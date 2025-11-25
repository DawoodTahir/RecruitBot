from __future__ import annotations

import asyncio
import json
import os
import pathlib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _create_openai_client():
    try:
        openai_module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenAI SDK is required. Install it with `pip install openai`."
        ) from exc
    return openai_module.OpenAI()


def _create_neo4j_graph(*, url: str, username: str, password: str):
    try:
        graphs_module = import_module("langchain_community.graphs")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langchain-community is required. Install it with `pip install langchain-community neo4j`."
        ) from exc
    Neo4jGraph = getattr(graphs_module, "Neo4jGraph")
    return Neo4jGraph(url=url, username=username, password=password)


def _create_neo4j_chain(graph, model_name: str):
    try:
        chain_module = import_module("langchain.chains.graph_qa.neo4j_graph")
        openai_module = import_module("langchain_openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langchain and langchain-openai are required. Install them with "
            "`pip install langchain langchain-openai`."
        ) from exc

    Neo4jGraphQAChain = getattr(chain_module, "Neo4jGraphQAChain")
    ChatOpenAI = getattr(openai_module, "ChatOpenAI")
    llm = ChatOpenAI(model=model_name, temperature=0)
    return Neo4jGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        return_intermediate_steps=True,
    )


@dataclass
class GraphQueryResult:
    answer: str
    intermediate_steps: list[Any]


class MCPClient:
    """
    Lightweight helper that spawns the MCP server and lets the agent call tools.
    """

    def __init__(self) -> None:
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []

    async def connect_to_server(self, server_script_path: str) -> None:
        server_params = StdioServerParameters(
            command="python",
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
        *,
        user_phone: str,
        message: str,
        tool_name: str = "notify_user_via_whatsapp",
    ) -> dict:
        if self.session is None:
            raise RuntimeError("MCP client is not connected to a server.")
        params = {"user_phone": user_phone, "message": message}
        return await self.session.call_tool(tool_name, params)

    async def close(self) -> None:
        await self.exit_stack.aclose()


class GraphRAG:
    """
    Graph RAG implementation backed by Neo4j + LangChain's Neo4jGraphQAChain.
    """

    def __init__(
        self,
        *,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        graph_llm_model: str = "gpt-4o-mini",
        min_keyword_overlap: int = 3,
    ) -> None:
        self.neo4j_uri = neo4j_uri or os.environ.get(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_username = neo4j_username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.min_keyword_overlap = min_keyword_overlap

        self.graph = _create_neo4j_graph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )
        self.chain = _create_neo4j_chain(self.graph, graph_llm_model)
        self._indexed = self._graph_has_content()

    def has_index(self) -> bool:
        return self._indexed

    async def index_document(self, path: str) -> None:
        path_obj = pathlib.Path(path).expanduser().resolve()
        text = await asyncio.to_thread(path_obj.read_text, encoding="utf-8")
        sections = await asyncio.to_thread(self._split_into_sections, text)
        await asyncio.to_thread(self._write_sections_to_graph, str(path_obj), sections)
        self._indexed = True

    async def query(self, question: str) -> Optional["GraphQueryResult"]:
        if not self._indexed:
            return None

        def _run_chain() -> "GraphQueryResult":
            result = self.chain.invoke({"query": question})
            answer = result.get("result", "")
            steps = result.get("intermediate_steps", [])
            return GraphQueryResult(answer=answer, intermediate_steps=steps)

        return await asyncio.to_thread(_run_chain)

    # ---- internal helpers -------------------------------------------------

    def _graph_has_content(self) -> bool:
        try:
            response = self.graph.query("MATCH (d:Document) RETURN count(d) AS count")
        except Exception:
            return False
        return bool(response and response[0].get("count", 0) > 0)

    def _split_into_sections(self, text: str) -> list[dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections: list[dict[str, Any]] = []
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

    def _write_sections_to_graph(self, doc_id: str, sections: list[dict[str, Any]]) -> None:
        if not sections:
            return

        self.graph.query(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.updatedAt = datetime(), d.sectionCount = size($sections)
            WITH d
            UNWIND $sections AS section
            MERGE (p:Paragraph {doc_id: $doc_id, idx: section.index})
            SET p.text = section.text
            MERGE (d)-[:HAS_SECTION]->(p)
            WITH p, section
            FOREACH (kw IN section.keywords |
                MERGE (k:Keyword {value: kw})
                MERGE (p)-[:HAS_KEYWORD]->(k)
            )
            """,
            {"doc_id": doc_id, "sections": sections},
        )

        # Sequential edges
        self.graph.query(
            """
            MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(p)
            WITH p ORDER BY p.idx
            WITH collect(p) AS nodes
            UNWIND range(0, size(nodes) - 2) AS i
            WITH nodes[i] AS src, nodes[i+1] AS dst
            MERGE (src)-[:NEXT]->(dst)
            MERGE (dst)-[:PREV]->(src)
            """,
            {"doc_id": doc_id},
        )

        # Similarity edges based on shared keywords
        self.graph.query(
            """
            MATCH (p1:Paragraph {doc_id: $doc_id})-[:HAS_KEYWORD]->(k)<-[:HAS_KEYWORD]-(p2:Paragraph {doc_id: $doc_id})
            WHERE p1.idx < p2.idx
            WITH p1, p2, count(k) AS shared
            WHERE shared >= $min_shared
            MERGE (p1)-[r:SIMILAR]->(p2)
            SET r.sharedKeywords = shared
            """,
            {"doc_id": doc_id, "min_shared": self.min_keyword_overlap},
        )

    def _extract_keywords(self, text: str) -> set[str]:
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

    def __init__(self, mcp_client: MCPClient, rag: GraphRAG, model: str = "gpt-4.1-mini") -> None:
        self.mcp_client = mcp_client
        self.rag = rag
        self.model = model
        self.llm = _create_openai_client()
        # In-memory conversation histories keyed by user_id.
        # For production, persist this in a database or cache.
        self._histories: dict[str, list[dict[str, str]]] = {}

    async def _call_llm(self, system_prompt: str, user_content: str) -> str:
        def _complete() -> str:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=800,
            )
            return response.choices[0].message.content or ""

        return await asyncio.to_thread(_complete)

    async def answer_question(self, user_message: str) -> str:
        graph_result = None
        if self.rag.has_index():
            graph_result = await self.rag.query(user_message)

        if graph_result:
            return await self._answer_with_graph(user_message, graph_result)
        return await self._answer_without_graph(user_message)

    async def _answer_without_graph(self, user_message: str) -> str:
        system_prompt = (
            "You are a helpful assistant. "
            "Answer the user using your own knowledge when no graph context is available."
        )
        reply = await self._call_llm(system_prompt, user_message)
        return reply.strip()

    async def _answer_with_graph(
        self, user_message: str, graph_result: GraphQueryResult
    ) -> str:
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

    async def send_answer_to_whatsapp(self, phone: str, message: str) -> None:
        await self.mcp_client.send_whatsapp_message(user_phone=phone, message=message)

    async def handle_message(
        self,
        *,
        user_id: str,
        message: str,
        send_to_whatsapp: bool = False,
        phone: str | None = None,
    ) -> dict[str, Any]:
        """
        Single entrypoint for the app/React frontend.
        - Runs graph RAG + LLM to get an answer.
        - Optionally sends the answer to WhatsApp via MCP.
        - Tracks short-term conversation history in memory.
        """
        graph_result = None
        if self.rag.has_index():
            graph_result = await self.rag.query(message)

        if graph_result:
            answer = await self._answer_with_graph(message, graph_result)
            graph_used = True
        else:
            answer = await self._answer_without_graph(message)
            graph_used = False

        history = self._histories.setdefault(user_id, [])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        tool_calls: list[dict[str, Any]] = []
        if send_to_whatsapp and phone:
            await self.send_answer_to_whatsapp(phone, answer)
            tool_calls.append(
                {"tool": "notify_user_via_whatsapp", "target": phone}
            )

        return {
            "answer": answer,
            "graph_used": graph_used,
            "graph_answer": getattr(graph_result, "answer", None)
            if graph_result
            else None,
            "graph_steps": getattr(graph_result, "intermediate_steps", None)
            if graph_result
            else None,
            "tool_calls": tool_calls,
        }
    async def cli_loop(self) -> None:
        last_answer: Optional[str] = None
        print(
            "Commands:\n"
            "  /upload <path>  -> build Graph RAG index over a file\n"
            "  /send <phone>   -> send last answer to WhatsApp\n"
            "  /quit           -> exit\n"
        )

        while True:
            user_input = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if not user_input:
                continue

            if user_input.lower() in {"/quit", "/exit"}:
                break

            if user_input.startswith("/upload "):
                path = user_input[len("/upload ") :].strip()
                try:
                    await self.rag.index_document(path)
                    print(f"Indexed document at {path}.")
                except OSError as exc:
                    print(f"Failed to read {path}: {exc}")
                continue

            if user_input.startswith("/send "):
                if not last_answer:
                    print("No answer available to send.")
                    continue
                phone = user_input[len("/send ") :].strip()
                await self.send_answer_to_whatsapp(phone, last_answer)
                print("Sent last answer via WhatsApp.")
                continue

            answer = await self.answer_question(user_input)
            print(f"Agent: {answer}")
            last_answer = answer


async def main() -> None:
    server_path = str(pathlib.Path(__file__).with_name("server.py").resolve())
    mcp_client = MCPClient()
    await mcp_client.connect_to_server(server_path)

    rag = GraphRAG()
    agent = ChatAgent(mcp_client=mcp_client, rag=rag)

    try:
        await agent.cli_loop()
    finally:
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
