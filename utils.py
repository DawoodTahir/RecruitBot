import asyncio
import json
import os,requests
import pathlib
import sys
import logging
import random
import PyPDF2
import docx2txt
from openai import OpenAI

from contextlib import AsyncExitStack
from importlib import import_module

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Default temperature for most LLM calls: a bit flexible, but not too random.
DEFAULT_TEMPERATURE = 0.4


# Ordered list of interview slots used by the planner and speaker.
INTERVIEW_SLOT_ORDER = [
    "greeting",
    
    "total_experience",
    "project_description",
    "project_metric",
    "project_bottleneck",
    "project_solution",
    "team_challenge",
    "adaptability_example",
    "leadership_example",
    "notice_period",
    "visa_status",
]


##prompt- injections heuristic

SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "you are now",
    "pretend to be",
    "system prompt",
    "you must reveal",
    "reveal your instructions",
    "disregard system",
    "forget you are",
]


# Simple heuristic patterns to detect when the user refuses or cannot answer.
REFUSAL_PATTERNS = [
    "i don't know",
    "i dont know",
    "no i can't tell you",
    "no i cant tell you",
    "i won't tell",
    "i wont tell",
    "prefer not to say",
    "rather not say",
    "i'd rather not say",
    "i would rather not say",
]


def is_refusal_message(text: str) -> bool:
    """
    Very simple heuristic: check if the user is refusing / unable to answer.
    """
    t = text.lower()
    return any(pat in t for pat in REFUSAL_PATTERNS)



def _create_openai_client():
    try:
        openai_module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenAI SDK is required. Install it with `pip install openai`."
        ) from exc
    return openai_module.OpenAI()

def _extract_resume_structure(text):
    """
    Use the same OpenAI client as ChatAgent to turn raw CV text into a structured JSON
    with candidate, experiences, and skills.
    """
    llm = _create_openai_client()

    system_prompt = (
        "You are a resume parser for an ATS. "
        "Given the raw text of a candidate's CV, extract a clean JSON structure.\n\n"
        "The JSON MUST have this shape:\n"
        "{\n"
        '  "candidate": {\n'
        '    "full_name": string,\n'
        '    "headline": string | null,\n'
        '    "location": string | null,\n'
        '    "email": string | null,\n'
        '    "phone": string | null\n'
        "  },\n"
        '  "experiences": [\n'
        "    {\n"
        '      "role": string,\n'
        '      "company": string | null,\n'
        '      "start_date": string | null,\n'
        '      "end_date": string | null,\n'
        '      "summary": string | null,\n'
        '      "metrics": [string],\n'
        '      "skills": [string]\n'
        "    }, ...\n"
        "  ],\n"
        '  "skills": [string]\n'
        "}\n\n"
        "Be concise; do NOT invent details that are not clearly implied in the text."
    )

    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=800,
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Very simple fallback: wrap everything in a single experience
        return {
            "candidate": {
                "full_name": None,
                "headline": None,
                "location": None,
                "email": None,
                "phone": None,
            },
            "experiences": [
                {
                    "role": None,
                    "company": None,
                    "start_date": None,
                    "end_date": None,
                    "summary": text[:4000],
                    "metrics": [],
                    "skills": [],
                }
            ],
            "skills": [],
        }
    return data

def _create_neo4j_graph(*, url, username, password):
    """
    Create a Neo4jGraph using the modern langchain-neo4j package when available.
    Fall back to None (disabling graph mode) if the dependency is missing.
    """
    try:
        neo4j_module = import_module("langchain_neo4j")
    except ModuleNotFoundError as exc:
        logging.warning(
            "langchain-neo4j / Neo4jGraph not available: %s. "
            "Graph RAG will run in in-memory fallback mode.",
            exc,
        )
        return None

    Neo4jGraph = getattr(neo4j_module, "Neo4jGraph", None)
    if Neo4jGraph is None:
        logging.warning(
            "Neo4jGraph class not found in langchain_neo4j; "
            "Graph RAG will run in in-memory fallback mode."
        )
        return None

    return Neo4jGraph(url=url, username=username, password=password)


def _create_neo4j_chain(graph, model_name):
    """
    Placeholder for a future graph QA chain.

    Newer versions of LangChain have significantly changed their chain APIs
    and no longer expose the legacy `langchain.chains.graph_qa.neo4j_graph`
    module. To keep this project stable across versions without pinning a very
    old LangChain, we avoid depending on a specific QA chain implementation
    here.

    For now this returns None, which means GraphRAG will still index resumes
    into Neo4j but will use the in-memory text fallback for answering queries.
    If you later want full graph QA, you can plug in a custom chain here that
    calls your LLM with Cypher results from `graph.query(...)`.
    """
    if graph is None:
        return None
        return None


class GraphQueryResult:
    def __init__(self, answer, intermediate_steps):
        self.answer = answer
        self.intermediate_steps = intermediate_steps


class InterviewState:
    """
    Holds structured information we want to collect during an interview-style flow.
    """

    def __init__(self, slots=None, goal_completed=False, ended=False, greeted=False):
        self.slots = slots if slots is not None else {}
        self.goal_completed = goal_completed
        self.ended = ended
        # Track whether we've already greeted the candidate by name.
        self.greeted = greeted


class PlannerDecision:
    """
    Output of the planner-level model: what should happen next in the interview.
    """

    def __init__(self, next_action, target_slot, updated_slots, goal_completed):
        self.next_action = next_action  # "ASK_SLOT" | "END"
        self.target_slot = target_slot
        self.updated_slots = updated_slots
        self.goal_completed = goal_completed

class MCPClient:
    """
    Lightweight helper that spawns the MCP server and lets the agent call tools.
    """

    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []

    async def connect_to_server(self, server_script_path):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools = [tool.name for tool in response.tools]
        print("Connected. Tools available from server:", self.available_tools)

    async def send_whatsapp_message(
        self,
        message,
        tool_name="notify_user_via_whatsapp",
    ):
        if self.session is None:
            raise RuntimeError("MCP client is not connected to a server.")
        params = {"message": message}
        return await self.session.call_tool(tool_name, params)

    async def close(self):
        await self.exit_stack.aclose()


class CompanyProfile:
    """
    Lightweight container for company insights used by the suggestion box.
    """

    def __init__(self, name, website, linkedin, services_summary, culture_summary, raw_snippets):
        self.name = name
        self.website = website
        self.linkedin = linkedin
        self.services_summary = services_summary
        self.culture_summary = culture_summary
        self.raw_snippets = raw_snippets


class CompanyInsightsScraper:
    """
    Scrapes a company's public website (and optionally LinkedIn) and uses an LLM
    to build a short profile: services + culture.
    """

    def __init__(self, openai_client=None, user_agent: str = "RecruitLensBot/1.0"):
        self.client = openai_client or _create_openai_client()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    async def scrape_company(
        self,
        *,
        company_name: str,
        website_url: str | None = None,
        linkedin_url: str | None = None,
        max_pages: int = 5,
    ) -> "CompanyProfile":
        """
        High-level entrypoint:
        - Crawl the website (if provided)
        - Optionally fetch LinkedIn summary text (stubbed)
        - Ask the LLM to summarise services + culture
        """
        snippets: list[str] = []

        if website_url:
            snippets.extend(await self._scrape_site(website_url, max_pages=max_pages))

        if linkedin_url:
            linkedin_text = await self._fetch_linkedin_public_summary(linkedin_url)
            if linkedin_text:
                snippets.append(linkedin_text)

        merged_text = "\n\n".join(snippets[:20])  # keep token size sane
        services_summary, culture_summary = await self._summarise_with_llm(
            company_name=company_name,
            website_url=website_url,
            linkedin_url=linkedin_url,
            text=merged_text,
        )

        return CompanyProfile(
            name=company_name,
            website=website_url,
            linkedin=linkedin_url,
            services_summary=services_summary,
            culture_summary=culture_summary,
            raw_snippets=snippets,
        )

    async def _scrape_site(self, base_url: str, max_pages: int = 5) -> list[str]:
        """
        Very simple breadth-first crawl limited to a few internal pages.
        Focuses on obvious 'About/Services/Careers' URLs.
        """
        import re
        import time
        from urllib.parse import urljoin, urlparse

        to_visit: list[str] = [base_url]
        visited: set[str] = set()
        texts: list[str] = []

        def is_same_domain(url: str) -> bool:
            try:
                base = urlparse(base_url).netloc
                netloc = urlparse(url).netloc
                return not netloc or netloc == base
            except Exception:
                return False

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = self.session.get(url, timeout=8)
                resp.raise_for_status()
            except Exception:
                continue

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "html.parser")

            for script in soup(["script", "style", "noscript"]):
                script.extract()
            text = " ".join(soup.get_text(separator=" ").split())
            if len(text) > 200:
                texts.append(text[:5000])

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "mailto:", "tel:")):
                    continue
                full = urljoin(base_url, href)
                if not is_same_domain(full):
                    continue
                if full in visited or full in to_visit:
                    continue
                if re.search(r"(about|team|company|culture|careers|service|product)", full, re.I):
                    to_visit.append(full)

            time.sleep(0.3)

        return texts

    async def _fetch_linkedin_public_summary(self, linkedin_url: str) -> str:
        """
        Placeholder for LinkedIn integration. For now this returns an empty string.
        In production, integrate with an approved LinkedIn API or accept pasted text.
        """
        return ""

    async def _summarise_with_llm(
        self,
        *,
        company_name: str,
        website_url: str | None,
        linkedin_url: str | None,
        text: str,
    ) -> tuple[str, str]:
        """
        Ask the LLM to produce two short summaries:
        - What the company does (services/products)
        - What their culture/employer brand looks like
        """
        if not text:
            return "", ""

        system_prompt = (
            "You are an expert B2B research analyst and HR advisor.\n"
            "You will receive raw text scraped from a company's website (and optionally LinkedIn).\n"
            "Your job is to build a concise briefing for a candidate who is about to interview there.\n\n"
            "1) SERVICES / BUSINESS SUMMARY:\n"
            "- What do they actually do? Products, services, target customers, markets.\n"
            "- Keep this factual and concrete.\n\n"
            "2) CULTURE / EMPLOYER BRAND SUMMARY:\n"
            "- Tone of their messaging, values, diversity/inclusion signals, how they talk about teams.\n"
            "- Any hints about how promising or people‑centric their culture is.\n\n"
            "Output JSON ONLY in this shape:\n"
            "{\n"
            '  \"services\": \"...\",\n'
            '  \"culture\": \"...\"\n'
            "}\n"
        )

        user_payload = {
            "company_name": company_name,
            "website_url": website_url,
            "linkedin_url": linkedin_url,
            "raw_text": text[:12000],
        }

        def _complete() -> tuple[str, str]:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or "{}"
            try:
                data = json.loads(content)
                return (
                    (data.get("services") or "").strip(),
                    (data.get("culture") or "").strip(),
                )
            except Exception:
                return "", ""

        return await asyncio.to_thread(_complete)
class Transcriber:
    """
    Handles Speech-to-Text using OpenAI Whisper and extracts basic prosodic features.
    """
    def __init__(self):
        self.client = _create_openai_client()

    def transcribe(self, audio_path):
        """
        Transcribes audio file and returns text + metrics (wpm, duration).
        """
        try:
            with open(audio_path, "rb") as audio_file:
                # Use verbose_json to get segment details if needed in future
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            text = transcript.text
            duration = getattr(transcript, "duration", 0)
            
            # Basic WPM calculation (Speech Rate)
            # Standard conversational English is ~130-150 WPM.
            # < 110 might indicate hesitation/thinking. > 160 might indicate anxiety/rushing.
            word_count = len(text.split())
            wpm = (word_count / duration * 60) if duration > 0 else 0
            
            return {
                "text": text,
                "metadata": {
                    "duration_seconds": duration,
                    "wpm": int(wpm),
                    "word_count": word_count
                }
            }
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return {"text": "", "metadata": {}}


class SoftSkillsAnalyzer:
    """
    Analyzes conversation history to extract OCEAN traits, soft skills, and PROSODY using a single LLM pass.
    """
    def __init__(self, model="gpt-4.1-mini"):
        self.llm = _create_openai_client()
        self.model = model

    async def generate_profile(self, conversation_history):
        """
        conversation_history: list of {"role": "user/assistant", "content": "...", "metadata": {...}}
        """
        # Filter only user messages
        user_msgs = [msg for msg in conversation_history if msg['role'] == 'user']
        
        # 1. Aggregate Audio Metrics (if available)
        total_wpm = 0
        wpm_count = 0
        for msg in user_msgs:
            meta = msg.get("metadata", {})
            if meta.get("wpm"):
                total_wpm += meta["wpm"]
                wpm_count += 1
        
        avg_wpm = int(total_wpm / wpm_count) if wpm_count > 0 else "N/A"
        
        # 2. Build Transcript
        user_text = "\n".join([f"- {msg['content']}" for msg in user_msgs])

        if not user_text:
            return "{}"

        system_prompt = (
            "You are an Expert Psycholinguist and Recruitment Psychologist. "
            "Analyze the candidate's conversation transcript to build a Soft Skills & Personality Profile.\n\n"
            f"**AUDIO METRICS (Measured from speech):** Average Speech Rate: {avg_wpm} WPM "
            "(Normal: 130-150. Low: <110, High: >160).\n\n"
            "## ANALYSIS FRAMEWORK\n"
            "1. STYLOMETRY (Linguistic Analysis):\n"
            "   - Pronoun Usage: High 'I/Me' (Individualist) vs 'We/Us' (Collaborative).\n"
            "   - Complexity: Sentence structure and vocabulary richness (Cognitive Ability).\n"
            "2. PROSODY & SPEECH MARKERS:\n"
            "   - Use the WPM metric above. Low WPM may indicate thoughtfulness OR hesitation. High WPM may indicate passion OR anxiety. Correlate this with the TEXT content.\n"
            "   - Search for dysfluencies in text ('um', 'uh', 'like').\n"
            "3. OCEAN (Big Five) SCORING (1-10 Scale):\n"
            "   - Openness: Curiosity, distinct vocabulary.\n"
            "   - Conscientiousness: Structure, logical flow.\n"
            "   - Extraversion: Energy, assertiveness.\n"
            "   - Agreeableness: Politeness, acknowledgement of others.\n"
            "   - Neuroticism (Emotional Stability): Calmness vs. defensive language.\n\n"
            "## OUTPUT FORMAT (JSON)\n"
            "Return a strictly valid JSON object (no markdown) with keys:\n"
            "{\n"
            "  \"ocean_scores\": {\"O\": int, \"C\": int, \"E\": int, \"A\": int, \"N\": int},\n"
            "  \"prosody_analysis\": {\"confidence\": \"High/Medium/Low\", \"fluency_notes\": string, \"avg_wpm\": int/string},\n"
            "  \"top_soft_skills\": [string],\n"
            "  \"communication_style\": string,\n"
            "  \"red_flags\": [string]\n"
            "}"
        )

        def _call_ai():
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"TRANSCRIPT:\n{user_text}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error generating soft skills profile: {e}")
                return "{}"

        return await asyncio.to_thread(_call_ai)

class GraphRAG:
    """
    Graph RAG that uses Neo4j + LangChain when available,
    and falls back to a simple in-memory text store otherwise.
    """

    def __init__(
        self,
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        graph_llm_model="gpt-4o-mini",
        min_keyword_overlap=3,
    ):
        # In-memory fallback state
        self._text = ""
        self._indexed = False
        # Parsed candidate info and full resume structure from the latest indexed resume (if any)
        self.candidate_info = {}
        self.resume_struct = {}

        # Neo4j / LangChain state
        self.neo4j_uri = neo4j_uri or os.environ.get(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_username = neo4j_username or os.environ.get(
            "NEO4J_USERNAME", "neo4j"
        )
        self.neo4j_password = neo4j_password or os.environ.get(
            "NEO4J_PASSWORD", "neo4j"
        )
        self.min_keyword_overlap = min_keyword_overlap

        # Try to set up Neo4j graph + chain. If anything fails, we just
        # operate in fallback mode without raising at import time.
        self.graph = None
        self.chain = None
        self._use_graph = False
        try:
            graph = _create_neo4j_graph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
            )
            chain = _create_neo4j_chain(graph, graph_llm_model)
            if graph is not None and chain is not None:
                self.graph = graph
                self.chain = chain
                self._use_graph = True
                self._indexed = self._graph_has_content()
                logging.info("GraphRAG: Neo4j graph mode enabled.")
            else:
                logging.info("GraphRAG: falling back to in-memory mode (no Neo4j).")
        except Exception as exc:
            logging.warning("GraphRAG: disabling Neo4j mode due to error: %s", exc)
            self.graph = None
            self.chain = None
            self._use_graph = False
            self._indexed = False

    def has_index(self):
        return self._indexed

    async def index_document(self, path, user_id: str):
        """
        Index a resume:
        - Read the file text
        - Store raw text for fallback RAG
        - If Neo4j + LangChain are available, extract resume structure and write to graph
        """
        path_obj = pathlib.Path(path).expanduser().resolve()
        text = await asyncio.to_thread(path_obj.read_text, encoding="utf-8")
        # Always keep raw text and parsed candidate info, regardless of graph mode.
        self._text = text
        resume_struct = _extract_resume_structure(text)
        self.resume_struct = resume_struct or {}
        self.candidate_info = resume_struct.get("candidate") or {}

        if self._use_graph and self.graph is not None:
            await asyncio.to_thread(
                self._write_resume_to_graph, user_id, resume_struct
            )

        # Mark that we have an index (either in-memory only or with graph)
        self._indexed = True

    async def query(self, question):
        if not self._indexed:
            return None

        if self._use_graph and self.chain is not None:
            def _run_chain():
                result = self.chain.invoke({"query": question})
                answer = result.get("result", "")
                steps = result.get("intermediate_steps", [])
                return GraphQueryResult(answer=answer, intermediate_steps=steps)

            return await asyncio.to_thread(_run_chain)

        # Fallback: return the raw text as context
        return GraphQueryResult(
            answer=self._text,
            intermediate_steps=[],
        )

    # ---- internal helpers for Neo4j mode -----------------------------------

    def _graph_has_content(self):
        if not self.graph:
            return False
        try:
            response = self.graph.query("MATCH (d:Document) RETURN count(d) AS count")
        except Exception:
            return False
        return bool(response and response[0].get("count", 0) > 0)

    def _split_into_sections(self, text):
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = []
        for idx, paragraph in enumerate(paragraphs):
            keywords = self._extract_keywords(paragraph)
            sections.append(
                {
                    "index": idx,
                    "text": paragraph,
                    "keywords": sorted(keywords),
                }
            )
        return sections

    def _write_resume_to_graph(self, candidate_id, resume_struct):
            """
            Write candidate, experiences, and skills into Neo4j using the schema:
            (:Candidate)-[:HAS_EXPERIENCE]->(:Experience)-[:USED_SKILL]->(:Skill)
            and (:Candidate)-[:HAS_SKILL]->(:Skill).
            """
            if not self.graph:
                return

            candidate = resume_struct.get("candidate") or {}
            experiences = resume_struct.get("experiences") or []
            skills = resume_struct.get("skills") or []

            # Upsert candidate node
            self.graph.query(
                """
                MERGE (c:Candidate {id: $cid})
                SET c.full_name = $full_name,
                    c.headline = $headline,
                    c.location = $location,
                    c.email = $email,
                    c.phone = $phone
                """,
                {
                    "cid": candidate_id,
                    "full_name": candidate.get("full_name"),
                    "headline": candidate.get("headline"),
                    "location": candidate.get("location"),
                    "email": candidate.get("email"),
                    "phone": candidate.get("phone"),
                },
            )

            # Experiences and their skills/metrics
            self.graph.query(
                """
                MATCH (c:Candidate {id: $cid})
                WITH c, $experiences AS exps
                UNWIND range(0, size(exps) - 1) AS i
                WITH c, exps[i] AS exp, i
                MERGE (e:Experience {candidate_id: $cid, idx: i})
                SET e.role = exp.role,
                    e.company = exp.company,
                    e.start_date = exp.start_date,
                    e.end_date = exp.end_date,
                    e.summary = exp.summary
                MERGE (c)-[:HAS_EXPERIENCE]->(e)
                """,
                {"cid": candidate_id, "experiences": experiences},
            )

            # Attach metrics to experiences
            self.graph.query(
                """
                MATCH (e:Experience {candidate_id: $cid})
                WITH e, $experiences AS exps
                UNWIND exps AS exp
                WITH e, exp
                WHERE e.idx = exp.idx
                UNWIND exp.metrics AS metric
                MERGE (m:Metric {value: metric})
                MERGE (e)-[:MEASURED_BY]->(m)
                """,
                {"cid": candidate_id, "experiences": [
                    dict(exp, idx=i) for i, exp in enumerate(experiences)
                ]},
            )

            # Skills for experiences and candidate
            self.graph.query(
                """
                MATCH (c:Candidate {id: $cid})
                WITH c, $experiences AS exps
                UNWIND exps AS exp
                UNWIND exp.skills AS skillName
                MERGE (s:Skill {name: skillName})
                WITH c, s, exp
                MERGE (c)-[:HAS_SKILL]->(s)
                WITH s, exp
                MATCH (e:Experience {candidate_id: $cid, idx: exp.idx})
                MERGE (e)-[:USED_SKILL]->(s)
                """,
                {"cid": candidate_id, "experiences": [
                    dict(exp, idx=i) for i, exp in enumerate(experiences)
                ]},
            )

            # Top-level skills list
            if skills:
                self.graph.query(
                    """
                    MATCH (c:Candidate {id: $cid})
                    UNWIND $skills AS skillName
                    MERGE (s:Skill {name: skillName})
                    MERGE (c)-[:HAS_SKILL]->(s)
                    """,
                    {"cid": candidate_id, "skills": skills},
                )


    def _extract_keywords(self, text):
        stopwords = {
            "the",
            "and",
            "or",
            "is",
            "are",
            "of",
            "to",
            "a",
            "in",
            "for",
            "on",
            "with",
            "that",
            "this",
            "it",
            "as",
            "by",
        }
        tokens = [
            token
            for token in "".join(
                ch.lower() if ch.isalnum() else " " for ch in text
            ).split()
            if len(token) > 2 and token not in stopwords
        ]
        return set(tokens)


class ChatAgent:
    """
    Orchestrates LLM calls, optional Graph RAG context, and MCP tool usage.
    """

    def __init__(self, mcp_client, rag, model="gpt-4.1-mini"):
        self.mcp_client = mcp_client
        self.rag = rag
        self.model = model
        self.llm = _create_openai_client()
        # In-memory conversation histories keyed by user_id.
        # For production, persist this in a database or cache.
        self._histories = {}
        # Per-user interview states for goal-oriented flows.
        self._interviews = {}
        #
        self.analyzer = SoftSkillsAnalyzer()
        self._owner_phone = "owner"
        # Audio Transcriber
        self.transcriber = Transcriber()

    async def generate_role_tech_keywords(self, role: str, limit: int = 15) -> list[str]:
        """
        For the side suggestion panel:
        Given a job title from the CV, return up to `limit` two-word (or very short)
        tech stack / library / methods keywords that are worth learning for this role.
        """
        system_prompt = (
            "You are a domain expert and career coach.\n"
            "Given a target role title, list the most important, in-demand skills and tech stacks "
            "for that role TODAY.\n"
            "Focus on concrete tools, libraries, frameworks, methods and technical competency areas "
            "that a candidate should learn or deepen.\n"
            "Each keyword should be at most TWO words (e.g. 'Finite Elements', 'CAD Automation', "
            "'Packaging Materials'). Avoid long sentences.\n"
            "Return ONLY valid JSON in this shape:\n"
            "{ \"skills\": [\"keyword 1\", \"keyword 2\", ...] }\n"
        )

        payload = {"role": role, "max_skills": limit}

        def _complete() -> str:
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=DEFAULT_TEMPERATURE,
            )
            return resp.choices[0].message.content or "{}"

        try:
            content = await asyncio.to_thread(_complete)
            data = json.loads(content)
            skills_list = data.get("skills") or []
            skills: list[str] = [
                str(s).strip() for s in skills_list if isinstance(s, str) and str(s).strip()
            ][:limit]
            return skills
        except Exception:
            return []

    async def generate_attitude_tips(self, role: str, skills: list[str]) -> list[str]:
        """
        Generate short two-word interview mindset / attitude tips for the given role+skills,
        used in the suggestion sidebar.
        """
        system_prompt = (
            "You are a senior hiring coach for high‑stakes technical interviews.\n"
            "Given a target role and a list of hot skills in that domain, produce a set of very short, "
            "two‑word attitude/mindset tips that help the candidate show up at their best.\n"
            "Rules for each tip:\n"
            "- It must be exactly TWO words (e.g. 'Stay calm', 'Trust yourself', 'Own outcomes').\n"
            "- Focus on confidence, clarity, listening, curiosity, and professionalism.\n"
            "- Do NOT mention specific tools, company names, or long phrases.\n"
            "Return ONLY valid JSON in this exact shape:\n"
            "{ \"tips\": [\"Two words\", \"Another tip\", ...] }\n"
        )

        payload = {"role": role, "skills": skills}

        def _complete() -> str:
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=DEFAULT_TEMPERATURE,
            )
            return resp.choices[0].message.content or "{}"

        try:
            content = await asyncio.to_thread(_complete)
            data = json.loads(content)
            tips_list = data.get("tips") or []
            return [str(t).strip() for t in tips_list if isinstance(t, str) and str(t).strip()]
        except Exception:
            return []

    def _maybe_prefix_greeting(self, state: InterviewState, answer: str) -> str:
        """
        Ensure we initiate the conversation with a friendly greeting (using the
        candidate's name when available) exactly once per interview.
        """
        if getattr(state, "greeted", False):
            return answer

        name = None
        if state.slots.get("full_name"):
            name = state.slots["full_name"]
        else:
            try:
                candidate = getattr(self.rag, "candidate_info", {}) or {}
                name = candidate.get("full_name")
            except Exception:
                name = None

        if name:
            preface = (
                f"Hi {name}, I'm RecruitLens. I'm here to walk through your experience "
                "and understand how you fit the target role."
            )
        else:
            preface = (
                "Hi there, I'm RecruitLens. I'm here to walk through your experience and "
                "understand how you fit the target role."
            )

        state.greeted = True
        state.slots["greeting"] = "completed"
        if answer.strip():
            return f"{preface}\n\n{answer}"
        return preface

    async def _call_llm(self, system_prompt, user_content):
        def _complete() -> str:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=800,
                temperature=DEFAULT_TEMPERATURE,
            )
            return response.choices[0].message.content or ""

        return await asyncio.to_thread(_complete)

    async def _call_planner(
        self, user_id, user_message, state, history
    ):
        """
        Planner-level call: given the current interview state and latest user message,
        decide which slot to ask about next or whether to end.
        """
        required_slots = {
            "greeting": "Internal flag to ensure the bot greets the candidate warmly using their name and CV context (no user input required).",
            "project_description": "A concrete description of one important project the candidate worked on.",
            "project_metric": "What metric or KPI they used to evaluate that project's success.",
            "project_bottleneck": "The main bottleneck or hardest part of that project.",
            "project_solution": "How they tackled and resolved that bottleneck.",
            "team_challenge": "A specific time they faced a challenge with a team member or stakeholder.",
            "adaptability_example": "A time they had to learn something new quickly or pivot their strategy.",
            "leadership_example": "An example of when they took ownership or led an initiative (even if not a manager).",
            "notice_period": "Their current notice period or when they can realistically start.",
            "visa_status": "Their current visa or work authorization status relevant to the job.",
        }

        cv_context = self.rag._text if self.rag and self.rag.has_index() else ""
        resume_struct = getattr(self.rag, "resume_struct", {}) if self.rag else {}
        job_requirements = os.environ.get("JOB_REQUIREMENTS", "")
        # Keep a small recent window of turns to inform behaviour without bloating the prompt.
        recent_history = history[-8:] if history else []

        system_prompt = (
            "You are an interview planner agent for a technical hiring bot.\n"
            "Your ONLY job is to manage a structured interview flow and output JSON describing "
            "what should happen next. You NEVER speak directly to the user.\n\n"
            "Interview mission:\n"
            "- Collect the candidate's key details in this order:\n"
            "  1) greeting (mark as completed automatically after the bot greets the candidate)\n"
            "  2) project_description\n"
            "  3) project_metric\n"
            "  4) project_bottleneck\n"
            "  5) project_solution\n"
            "  6) team_challenge\n"
            "  7) adaptability_example\n"
            "  8) leadership_example\n"
            "  9) notice_period\n"
            "  10) visa_status\n\n"
            "You are also given:\n"
            "- cv_context: raw or summarized text from the candidate's CV.\n"
            "- resume_struct: structured JSON parsed from the CV with candidate + experiences + skills.\n"
            "- job_requirements: text describing the role's requirements.\n"
            "You MUST base your choice of next slot on BOTH the candidate's CV (cv_context / resume_struct) "
            "and the job requirements, prioritizing questions that clarify how their past projects and skills "
            "match the job needs.\n\n"
            "Required slots and their meanings:\n"
            f"{json.dumps(required_slots, indent=2)}\n\n"
            "State object:\n"
            "- slots: mapping from slot name to its current value (or null if unknown)\n"
            "- goal_completed: whether all important information has been gathered\n"
            "- ended: whether the interview has already been closed\n\n"
            "Rules:\n"
            "1. If ended is true, always output next_action = \"END\" and goal_completed = true.\n"
            "2. Otherwise, if you can reliably fill any slots from the latest user message, "
            "   add them to updated_slots.\n"
            "3. When possible, infer project-related slots directly from resume_struct or cv_context, "
            "   instead of asking the user to repeat obvious CV facts.\n"
            "4. Use history and latest_user_message to avoid sounding robotic: if the user clearly refuses "
            "   or says they don't know (e.g. 'I can't tell you that', 'I don't know', 'prefer not to say'), "
            "   do NOT simply repeat the same question again. You may instead:\n"
            "   - mark that slot as effectively unknown in updated_slots (for example, 'UNKNOWN'), and pick a\n"
            "     different missing slot, or\n"
            "   - switch target_slot to a different high-value slot that moves the interview forward.\n"
            "5. If after updates all required slots are filled, set goal_completed = true and "
            "   next_action = \"END\".\n"
            "6. If some slots are still missing, set next_action = \"ASK_SLOT\" and target_slot "
            "   to the MOST appropriate missing slot, following the order above as a guideline.\n"
            "7. Do NOT ask about anything outside these slots (no company secrets, internal metrics, etc.).\n"
            "8. Never include natural language questions or greetings. Only output JSON.\n\n"
            "You must respond with a single JSON object of the form:\n"
            '{\n'
            '  \"next_action\": \"ASK_SLOT\" | \"END\",\n'
            '  \"target_slot\": string or null,\n'
            '  \"updated_slots\": { \"slot_name\": \"value\", ... },\n'
            '  \"goal_completed\": boolean\n'
            "}\n"
        )

        planner_input = {
            "user_id": user_id,
            "state": {
                "slots": state.slots,
                "goal_completed": state.goal_completed,
                "ended": state.ended,
            },
            "latest_user_message": user_message,
            "cv_context": cv_context,
            "resume_struct": resume_struct,
            "job_requirements": job_requirements,
            "history": recent_history,
        }

        def _complete():
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(planner_input)},
                ],
                max_tokens=600,
                temperature=DEFAULT_TEMPERATURE,
            )
            raw = response.choices[0].message.content or "{}"
            try:
                data = json.loads(raw)
            ## if it fails to get into json format
            except json.JSONDecodeError:
                # Fallback: simple default asking for name.
                return PlannerDecision(
                    next_action="ASK_SLOT",
                    target_slot="name",
                    updated_slots={},
                    goal_completed=False,
                )

            next_action = data.get("next_action", "ASK_SLOT")
            target_slot = data.get("target_slot")
            updated_slots = data.get("updated_slots") or {}
            goal_completed = bool(data.get("goal_completed", False))

            if not isinstance(updated_slots, dict):
                updated_slots = {}

            return PlannerDecision(
                next_action=str(next_action),
                target_slot=str(target_slot) if target_slot is not None else None,
                updated_slots={str(k): str(v) for k, v in updated_slots.items()},
                goal_completed=goal_completed,
            )

        return await asyncio.to_thread(_complete)

    def _get_or_create_interview_state(self, user_id):
        if user_id not in self._interviews:
            # Initialize slots aligned with planner expectations.
            slots = {
                "full_name": None,
                "total_experience": None,
                "project_description": None,
                "project_metric": None,
                "project_bottleneck": None,
                "project_solution": None,
                "team_challenge": None,
                "adaptability_example": None,
                "leadership_example": None,
                "notice_period": None,
                "visa_status": None,
            }
            # If we already parsed a resume, pre-fill what we can (e.g. full_name).
            try:
                candidate = getattr(self.rag, "candidate_info", {}) or {}
                resume_struct = getattr(self.rag, "resume_struct", {}) or {}
            except Exception:
                candidate = {}
                resume_struct = {}
            if candidate.get("full_name"):
                slots["full_name"] = candidate["full_name"]
            total_xp = (
                candidate.get("total_experience_years")
                or candidate.get("total_experience")
                or resume_struct.get("total_experience_years")
                or resume_struct.get("total_experience")
            )
            if total_xp:
                slots["total_experience"] = str(total_xp)

            self._interviews[user_id] = InterviewState(slots=slots)
        return self._interviews[user_id]

    async def answer_question(self, user_message):
        graph_result = await self.rag.query(user_message)

        if graph_result:
            return await self._answer_with_graph(user_message, graph_result)
        else:
            return (self._call_llm("tell me about your experience in domain", user_message)).strip()
    # async def _answer_without_graph(self, user_message):
    #     system_prompt = (
    #         "You are a helpful assistant. "
    #         "Answer the user using your own knowledge when no graph context is available."
    #     )
    #     reply = await self._call_llm(system_prompt, user_message)
    #     return reply.strip()

    async def _answer_with_graph(
        self, user_message, graph_result
    ):
        structured_context = json.dumps(
            {
                "graph_answer": graph_result.answer,
                "intermediate_steps": graph_result.intermediate_steps,
            },
            ensure_ascii=False,
            indent=2,
        )

        system_prompt = (
            "You are a graph-aware assistant. "
            "You are given (a) the graph query answer produced by a Neo4jGraphQAChain "
            "and (b) the intermediate traversal steps that show how the chain navigated the graph. "
            "Treat the graph answer as authoritative but verify it using the intermediate steps. "
            "If the steps look incomplete, explain the gap and suggest follow-up queries."
        )

        user_payload = (
            f"User question:\n{user_message}\n\n"
            f"Graph retrieval context (answer + steps):\n{structured_context}"
        )

        reply = await self._call_llm(system_prompt, user_payload)
        return reply.strip()

    async def send_answer_to_whatsapp(self, phone, message):
        await self.mcp_client.send_whatsapp_message(user_phone=phone, message=message)

    async def _speak_question_for_slot(
        self, target_slot, state, latest_user_message=None
    ):
        """
        Speaker-level helper to turn a planner-selected slot into a natural-language question.
        """
        system_prompt = (
            "You are an interviewer chatbot for a hiring process. "
            "You already know which specific information (slot) you need next.\n"
            "You are given:\n"
            "- target_slot: the field you must collect next (e.g. project_description, project_metric).\n"
            "- known_information: the current slot values.\n"
            "- resume_struct: structured CV JSON with candidate + experiences + skills.\n"
            "- cv_context: raw or summarized text from the candidate's CV.\n"
            "- job_requirements: text describing the role's requirements.\n"
            "- latest_user_message: what the candidate just said.\n\n"
            "Your job is to ask ONE clear, natural follow-up question to collect that specific piece "
            "of information from the candidate.\n\n"
            "When the slot is project-related (project_description, project_metric, project_bottleneck, "
            "project_solution, team_challenge, adaptability_example, leadership_example):\n"
            "- Ground the question in ONE concrete project from resume_struct.experiences.\n"
            "- Refer to the project explicitly by company or project name when possible.\n"
            "- Ask for specific details (what they did, metrics, bottlenecks, decisions, impact).\n\n"
            "For more generic slots (full_name, total_experience, notice_period, visa_status):\n"
            "- Keep questions short and factual.\n"
            "- You may briefly tie them back to the CV when it helps (e.g. \"across your last 3 roles...\").\n"
            "- For total_experience specifically, prefer a formulation like:\n"
            "  \"Across your roles at <company1> and <company2>, could you please confirm your total professional "
            "experience in years and months?\"\n"
            "  Use company names from resume_struct.experiences when available, and always explicitly ask for "
            "\"years and months\".\n\n"
            "If latest_user_message clearly refuses to answer or says they don't know (for example "
            "'I can't tell you that' or 'I don't know'), acknowledge that politely and either:\n"
            "- offer a softer way to answer (for example a rough estimate or a range), or\n"
            "- gently reframe the question so it feels less repetitive.\n"
            "Do NOT repeat exactly the same wording as a previous question.\n\n"
            "Be concise, friendly, and professional. Do NOT mention slot names or internal fields. "
            "Prefer questions that clarify how the candidate's past projects and skills match the job needs."
        )
        user_payload = json.dumps(
            {
                "target_slot": target_slot,
                "known_information": state.slots,
                "cv_context": self.rag._text if self.rag and self.rag.has_index() else "",
                "resume_struct": getattr(self.rag, "resume_struct", {}) if self.rag else {},
                "job_requirements": os.environ.get("JOB_REQUIREMENTS", ""),
                "latest_user_message": latest_user_message or "",
            },
            ensure_ascii=False,
        )
        reply = await self._call_llm(system_prompt, user_payload)
        return reply.strip()

    async def _speak_deepen_future_slot(self, slot_name, slot_value, state):
        """
        When the candidate has already answered a future slot earlier than expected,
        acknowledge that understanding and politely ask them to go a bit deeper.
        """
        system_prompt = (
            "You are an interviewer chatbot for a hiring process.\n"
            "The candidate has already provided an answer for a later interview slot earlier than expected.\n"
            "Your job is to:\n"
            "1) Briefly reaffirm that you understood what they said for that specific topic.\n"
            "2) Ask them to explain it in a bit more detail (one clear follow-up question).\n"
            "Do NOT mention internal slot names or ordering; stay natural and conversational.\n"
        )
        user_payload = json.dumps(
            {
                "slot_name": slot_name,
                "slot_value": slot_value,
                "known_information": state.slots,
                "cv_context": self.rag._text if self.rag and self.rag.has_index() else "",
                "job_requirements": os.environ.get("JOB_REQUIREMENTS", ""),
            },
            ensure_ascii=False,
        )
        reply = await self._call_llm(system_prompt, user_payload)
        return reply.strip()

    async def _speak_goodbye(self, state):
        """
        Speaker-level helper to generate a short closing message when the interview ends.
        """
        system_prompt = (
            "You are an interviewer chatbot. The interview is complete and you have "
            "collected the user's key details. Thank the user, optionally reflect "
            "briefly on what you learned, and say goodbye in a sentence or two."
        )
        user_payload = json.dumps({"slots": state.slots}, ensure_ascii=False)
        reply = await self._call_llm(system_prompt, user_payload)
        return reply.strip()

    async def _send_owner_report(self, user_id, state):
        """
        After an interview ends, produce a structured report and send it to the owner via WhatsApp.
        """
        history = self._histories.get(user_id, [])

        # 1. Run Soft Skills Analysis (OCEAN)
        soft_skills_json_str = await self.analyzer.generate_profile(history)
        
        try:
             soft_skills = json.loads(soft_skills_json_str)
             ocean = soft_skills.get("ocean_scores", {})
             ocean_str = ", ".join([f"{k}:{v}" for k, v in ocean.items()])
             
             prosody = soft_skills.get("prosody_analysis", {})
             prosody_str = f"Confidence: {prosody.get('confidence', 'N/A')}, Avg WPM: {prosody.get('avg_wpm', 'N/A')}"
             
             summary = f"Style: {soft_skills.get('communication_style', 'N/A')}"
             soft_skills_section = f"🧠 **Psychometric Profile**:\n- OCEAN: {ocean_str}\n- PROSODY: {prosody_str}\n- {summary}"
        except Exception:
             soft_skills_section = "🧠 **Psychometric Profile**: Analysis failed or insufficient data."

        if not self._owner_phone:
            return

        system_prompt = (
            "You are an assistant that writes concise structured reports for a human owner. "
            "Given the collected interview slots about a client, produce a short report "
            "summarizing who they are, what they need, their constraints, and any notable details. "
            "Write it as plain text, suitable to be sent in a single WhatsApp message."
        )
        user_payload = json.dumps(
            {"user_id": user_id, "slots": state.slots}, ensure_ascii=False, indent=2
        )
        report_text = await self._call_llm(system_prompt, user_payload)

        # 3. Combine them
        full_report = f"{report_text}\n\n{soft_skills_section}"

        await self.mcp_client.send_whatsapp_message(
            message=full_report.strip(),
        )

    async def start_interview(self, user_id):
        """
        Kick off the interview immediately after a resume has been uploaded.

        Behaviour:
        - assumes the resume is already indexed in self.rag
        - initialises / reuses the InterviewState (so slots like full_name are pre‑filled)
        - sends a *pure greeting / why‑we‑are‑here* message, without asking a slot question yet
        - lets the candidate say something first
        - the next user turn will go through handle_message(), which will start the routed interview
        """
        if not self.rag.has_index():
            return {
                "answer": "Please upload your latest resume.",
                "interview_state": None,
                "tool_calls": [],
                "flagged": False,
                "reason": "resume_missing",
            }

        history = self._histories.setdefault(user_id, [])
        state = self._get_or_create_interview_state(user_id)

        # For the initial kick‑off we only send a greeting / context message.
        # The first actual interview question will be generated after the user
        # replies, inside handle_message().
        answer = self._maybe_prefix_greeting(state, "")
        tool_calls: list[dict] = []
        next_input_mode = "text"

        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "interview_state": {
                "slots": state.slots,
                "goal_completed": state.goal_completed,
                "ended": state.ended,
            },
            "tool_calls": tool_calls,
            "next_input_mode": next_input_mode,
        }

    ##Entry point of the app
    async def handle_message(
        self,
        user_id,
        message,
        audio_metrics=None
    ):
        """
        Single entrypoint for the app/React frontend.
        Router logic:
        - Update interview state from the user's latest message (planner).
        - Decide whether to ask a new question or end the interview.
        - Optionally send the final answer to WhatsApp for the user.
        - After goal completion and goodbye, send a structured report to the owner.
        - Randomly require some answers (especially deeper project questions)
          to be provided via voice, so that the prosody/OCEAN analyser has signal.
        """
        if not self.rag.has_index():
            return {
                "answer": "Please upload your latest resume.",
                "interview_state": None,
                "tool_calls": [],
                "flagged": False,
                "reason": "resume_missing",
            }
        # Track raw history for context if needed later.
        history = self._histories.setdefault(user_id, [])

        # Attach metrics (WPM) to the message for SoftSkillsAnalyzer and
        # detect refusal-style replies as a behavioural signal.
        msg_obj = {"role": "user", "content": message}
        if audio_metrics:
            msg_obj["metadata"] = dict(audio_metrics)
        if is_refusal_message(message):
            meta = msg_obj.setdefault("metadata", {})
            meta["refusal"] = True
        history.append(msg_obj)

        state = self._get_or_create_interview_state(user_id)
        # Snapshot slots before planner updates so we can detect newly filled ones.
        previous_slots = dict(state.slots)

        decision = await self._call_planner(
            user_id=user_id, user_message=message, state=state, history=history
        )

        # Apply any slot updates from the planner.
        for key, value in decision.updated_slots.items():
            if key in state.slots:
                state.slots[key] = value
        # Greeting slot is a virtual slot; mark as done once we greet.
        if decision.target_slot == "greeting":
            state.greeted = True
            decision.updated_slots["greeting"] = "completed"

        state.goal_completed = decision.goal_completed

        tool_calls = []
        # Default: text input for next turn, unless we decide to force voice.
        next_input_mode = "text"

        if decision.next_action == "END":
            state.ended = True
            answer = await self._speak_goodbye(state)
            # After we say goodbye, prepare and send a report to the owner.
            await self._send_owner_report(user_id, state)
        else:
            # Planner decided we still need to ask about some slot.
            target_slot = decision.target_slot or "goals"
            if target_slot == "greeting":
                state.greeted = True
                target_slot = "project_description"
            answer = await self._speak_question_for_slot(
                target_slot=target_slot, state=state, latest_user_message=message
            )

            # If the candidate's last answer also filled any *future* slots in the
            # sequence, gently acknowledge and ask them to go a bit deeper on one.
            slot_index = {name: idx for idx, name in enumerate(INTERVIEW_SLOT_ORDER)}
            newly_filled = []
            for key, value in decision.updated_slots.items():
                if key not in state.slots:
                    continue
                before_val = previous_slots.get(key)
                after_val = state.slots.get(key)
                if (before_val is None or str(before_val).strip() == "") and after_val:
                    newly_filled.append(key)

            target_idx = slot_index.get(decision.target_slot, -1) if decision.target_slot else -1
            future_slots = [
                s for s in newly_filled if slot_index.get(s, len(INTERVIEW_SLOT_ORDER)) > target_idx
            ]
            if future_slots:
                future_slot = future_slots[0]
                future_value = state.slots.get(future_slot)
                if future_value:
                    deepen_text = await self._speak_deepen_future_slot(
                        slot_name=future_slot,
                        slot_value=future_value,
                        state=state,
                    )
                    # Combine reaffirmation + original next question.
                    answer = f"{deepen_text}\n\n{answer}"

            # For some project / behavioural slots, randomly require VOICE input next
            # so that we can collect prosodic features via the /voice endpoint.
            voice_eligible_slots = {
                "project_description",
                "project_metric",
                "project_bottleneck",
                "project_solution",
                "team_challenge",
                "adaptability_example",
                "leadership_example",
            }
            try:
                if (
                    decision.target_slot in voice_eligible_slots
                    and random.random() < 0.4  # ~40% of these will be voice-only
                ):
                    next_input_mode = "voice"
                    answer = (
                        "I need to hear this one — please tap the Voice button and tell me about it.\n\n"
                        f"{answer}"
                    )
            except Exception:
                next_input_mode = "text"

        # Greet the candidate by name once, if we have their name from the resume or slots.
        answer = self._maybe_prefix_greeting(state, answer)

        history.append({"role": "assistant", "content": answer})

        # if send_to_whatsapp and phone:
        #     await self.send_answer_to_whatsapp(phone, answer)
        #     tool_calls.append(
        #         {"tool": "notify_user_via_whatsapp", "target": phone}
        #     )

        return {
            "answer": answer,
            "interview_state": {
                "slots": state.slots,
                "goal_completed": state.goal_completed,
                "ended": state.ended,
            },
            "tool_calls": tool_calls,
            "next_input_mode": next_input_mode,
        }


class Preprocess:
    def init(self):
        pass

        
    def extract_pdf(self,pdf_path, txt_path):
        text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        with open(txt_path, "w", encoding="utf-8") as out:
            out.write("\n\n".join(text))

    def extract_docx_to_text(self,docx_path, txt_path):
        text = docx2txt.process(docx_path) or ""
        with open(txt_path, "w", encoding="utf-8") as out:
            out.write(text)

    def heuristic_prompt_injections(self,text):
        l_case = text.lower()
        for pat in SUSPICIOUS_PATTERNS:
            if pat in l_case:
                return True
        return False
    
    def classify_prompt_risk(self,text):
        """
        heuristic flag + small openai classifier to detect for the misuse
        """
        heuristic_flag = self.heuristic_prompt_injections(text)

        try:
            classifier = OpenAI()
            resp = classifier.moderations.create(
                model = "omni-moderation-latest",
                input=text
            )

            result = resp.results[0]
            harmful_flag = bool(getattr(result, "flagged", False))

            # Extract categories in a JSON‑serialisable form
            raw_categories = getattr(result, "categories", {})
            if isinstance(raw_categories, dict):
                categories = raw_categories
            elif hasattr(raw_categories, "model_dump"):
                # OpenAI objects (pydantic‑style) generally support model_dump()
                categories = raw_categories.model_dump()
            elif hasattr(raw_categories, "__dict__"):
                # Fallback: use the object's __dict__
                categories = dict(raw_categories.__dict__)
            else:
                # Last resort: stringify so Flask jsonify can handle it
                categories = str(raw_categories)
        except Exception as exc:
            harmful_flag = False
            categories = {}

        # Allowed only if neither heuristic nor model flags the content
        allowed = not (heuristic_flag or harmful_flag)

        return {
            "allowed" : allowed,
            "heuristic_flag" : heuristic_flag,
            "harmful_flag"  : harmful_flag,
            "categories"    : categories
        }    
