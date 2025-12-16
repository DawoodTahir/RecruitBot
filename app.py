import uuid, logging, json, time
from flask import Flask, request, jsonify, send_from_directory
import asyncio
import os
from pathlib import Path
from utils import MCPClient, GraphRAG, ChatAgent, Preprocess, CompanyInsightsScraper
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("chatbot")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(("%(message)s")))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok = True)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")  # e.g. http://localhost:4566 for LocalStack
S3_BUCKET_RESUMES = os.environ.get("S3_BUCKET_RESUMES", "recruitlens-resumes")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")  # optional; used in later phases for workers

_s3_client = None
_sqs_client = None

if AWS_ENDPOINT_URL:
    try:
        _s3_client = boto3.client("s3", region_name=AWS_REGION, endpoint_url=AWS_ENDPOINT_URL)
    except Exception as exc:
        logging.warning("Failed to create S3 client (endpoint %s): %s", AWS_ENDPOINT_URL, exc)
        _s3_client = None
    try:
        if SQS_QUEUE_URL:
            _sqs_client = boto3.client("sqs", region_name=AWS_REGION, endpoint_url=AWS_ENDPOINT_URL)
    except Exception as exc:
        logging.warning("Failed to create SQS client (endpoint %s): %s", AWS_ENDPOINT_URL, exc)
        _sqs_client = None
else:
    try:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    except Exception as exc:
        logging.warning("Failed to create S3 client: %s", exc)
        _s3_client = None
    try:
        if SQS_QUEUE_URL:
            _sqs_client = boto3.client("sqs", region_name=AWS_REGION)
    except Exception as exc:
        logging.warning("Failed to create SQS client: %s", exc)
        _sqs_client = None

def _upload_to_s3(local_path: Path, user_id: str, filename: str) -> str | None:
    
    if not _s3_client:
        return None
    key = f"{user_id}/{filename}"
    try:
        with open(local_path, "rb") as f:
            _s3_client.upload_fileobj(f, S3_BUCKET_RESUMES, key)
        return key
    except (BotoCoreError, ClientError, OSError) as exc:
        logging.warning("S3 upload failed for %s: %s", local_path, exc)
        return None

def _enqueue_resume_job(user_id: str, s3_key: str | None, ext: str) -> None:
   
    if not _sqs_client or not SQS_QUEUE_URL or not s3_key:
        return
    body = {
        "type": "resume_uploaded",
        "user_id": user_id,
        "s3_key": s3_key,
        "ext": ext,
    }
    try:
        _sqs_client.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(body))
    except (BotoCoreError, ClientError) as exc:
        logging.warning("Failed to enqueue resume job to SQS: %s", exc)


SUGGESTIONS_CACHE: dict[tuple[str, str], dict] = {}


##App initialize
app = Flask(__name__)

async def init():
    global agent , preprocess, company_scraper


    mcp_client = MCPClient()

    server = str(BASE_DIR / "server.py") ##add the path
    await mcp_client.connect_to_server(server)

    rag = GraphRAG()
    agent = ChatAgent(mcp_client=mcp_client,rag=rag)
    company_scraper = CompanyInsightsScraper()
    preprocess = Preprocess()





@app.route("/chat", methods = ["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_id = data.get("user_id","anonymous")
    message = data.get("message", "")
    

    request_id = str(uuid.uuid4())
    t0 = time.time()

    risk_service = preprocess
    risky_input = risk_service.classify_prompt_risk(message)
    if not risky_input["allowed"]:
        app.logger.warning(
            "Blocked user message for user %s: heuristic=%s harmful=%s cats=%s",
            user_id,
            risky_input["heuristic_flag"],
            risky_input["harmful_flag"],
            risky_input["categories"],
        )

        return jsonify({
            "answer": "I can’t follow those instructions, but I can still help with normal questions.",
            "interview_state": None,
            "tool_calls": [],
            "flagged": True,
            "reason": "input_blocked",
            "risk": risky_input,
        })
    

    async def _run():

        return  await agent.handle_message(
            user_id = user_id,
            message = message,
            
        )

    
    result= asyncio.run(_run())
    
    latency_ms = int((time.time() - t0) * 1000)
    output_risk = risk_service.classify_prompt_risk(result.get("answer", ""))
    if not output_risk["allowed"]:
        app.logger.warning(
            "Blocked output for user %s: heuristic=%s harmful=%s cats=%s",
            user_id,
            output_risk["heuristic_flag"],
            output_risk["harmful_flag"],
            output_risk["categories"],
        )
        result["answer"] = (
            "I'm not able to provide that response. Please try asking in a different way."
        )
        result["flagged"] = True
        result["reason"] = "output_blocked"
        result["risk"] = output_risk

    ##Logger json for the event happened with a specific id
    logger.info(json.dumps({
            "type": "request",
            "endpoint": "/chat",
            "request_id": request_id,
            "user_id": user_id,
            "latency_ms": latency_ms,
            "status": "ok",
            "graph_used": bool(result.get("interview_state")),  # or other flag
            "tool_calls": result.get("tool_calls", []),
        }))

    return jsonify(result)


@app.route("/voice", methods=["POST"])
def voice_chat():
    """
    Receives an audio file (blob) from the frontend.
    Transcribes it using OpenAI Whisper.
    Passes text + audio metrics to the ChatAgent.
    """
    request_id = str(uuid.uuid4())
    t0 = time.time()
    
    user_id = request.form.get("user_id", "anonymous")
    
    if "audio" not in request.files:
        return jsonify({"error": "no audio part"}), 400
        
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    # Save temp file
    filename = f"{uuid.uuid4()}.webm" # assume webm from browser or wav
    save_path = UPLOADS / filename
    audio_file.save(save_path)

    # 1. Transcribe
    transcription_result = agent.transcriber.transcribe(str(save_path))
    text = transcription_result.get("text", "")
    metrics = transcription_result.get("metadata", {})
    
    # Clean up temp file
    try:
        os.remove(save_path)
    except OSError:
        pass

    if not text:
        return jsonify({"answer": "I couldn't hear you clearly. Could you repeat that?", "tool_calls": []})

    # 2. Process with Agent (passing audio metrics)
    async def _run():
        return await agent.handle_message(
            user_id=user_id,
            message=text,
            audio_metrics=metrics # Pass WPM etc.
        )

    result = asyncio.run(_run())
    result["transcription"] = text # send back what we heard
    
    latency_ms = int((time.time() - t0) * 1000)
    logger.info(json.dumps({
        "type": "voice_request",
        "endpoint": "/voice",
        "user_id": user_id,
        "latency_ms": latency_ms,
        "wpm": metrics.get("wpm", 0)
    }))

    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    request_id = str(uuid.uuid4())
    t0 = time.time()
    user_id = request.form.get("user_id", "anonymous")
    if "file" not in request.files:
        logger.info(json.dumps({
            "type": "request",
            "endpoint": "/upload",
            "request_id": request_id,
            "status": "error",
            "error": "no_file_part",
        }))
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.info(json.dumps({
            "type": "request",
            "endpoint": "/upload",
            "request_id": request_id,
            "status": "error",
            "error": "empty_filename",
        }))
        return jsonify({"error":"no filename found"}), 400
    
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    save_path = UPLOADS / filename
    file.save(save_path)

    
    s3_key = _upload_to_s3(save_path, user_id=user_id, filename=filename)
   
    _enqueue_resume_job(user_id=user_id, s3_key=s3_key, ext=ext)

    payload = {
        "status": "ok",
        "indexed_path": None,
        "s3_key": s3_key,
        "resume_indexed": False,
    }
    return jsonify(payload)


@app.route("/suggestions", methods=["GET"])
def suggestions():
    
    user_id = request.args.get("user_id", "anonymous")
    explicit_role = request.args.get("role")

    role_title = explicit_role
    try:
        candidate = getattr(agent.rag, "candidate_info", {}) or {}
        resume_struct = getattr(agent.rag, "resume_struct", {}) or {}
    except Exception:
        candidate = {}
        resume_struct = {}

    if not role_title:
       
        headline = (candidate.get("headline") or "").strip()
        if headline and len(headline.split()) > 3:
            role_title = headline
        else:
            
            base = (
                (candidate.get("headline") or "").strip()
                or (candidate.get("title") or "").strip()
                or (candidate.get("role") or "").strip()
            )
            skills = resume_struct.get("skills") or []
            skills_fragment = ""
            if skills:
                top_skills = ", ".join(skills[:2])
                skills_fragment = f" with experience in {top_skills}"
            role_title = (base + skills_fragment).strip()

    if not role_title:

        role_title = "Machine Learning Engineer"

    occupation_label = role_title

   
    cache_key = (user_id, role_title)
    cached = SUGGESTIONS_CACHE.get(cache_key)
    if cached is not None:
        payload = {
            "role": role_title,
            "hot_skills": cached.get("hot_skills", []),
            "attitude_tips": cached.get("attitude_tips", []),
            "company": cached.get("company"),
        }
        return jsonify(payload)

    company_name = os.environ.get("COMPANY_NAME", "").strip()
    company_website = os.environ.get("COMPANY_WEBSITE", "").strip() or None
    company_linkedin = os.environ.get("COMPANY_LINKEDIN", "").strip() or None

    async def _run_tools():
        try:
            hot = await agent.generate_role_tech_keywords(role_title, limit=15)
        except Exception as exc:
            logger.warning("generate_role_tech_keywords failed: %s", exc)
            hot = []

        try:
            tips = await agent.generate_attitude_tips(occupation_label, hot)
        except Exception as exc:
            logger.warning("generate_attitude_tips failed: %s", exc)
            tips = []

        company_profile = None
        if company_name and company_website:
            try:
                profile = await company_scraper.scrape_company(
                    company_name=company_name,
                    website_url=company_website,
                    linkedin_url=company_linkedin,
                    max_pages=5,
                )
                company_profile = {
                    "name": profile.name,
                    "website": profile.website,
                    "linkedin": profile.linkedin,
                    "services_summary": profile.services_summary,
                    "culture_summary": profile.culture_summary,
                }
            except Exception as e:
                logger.warning("CompanyInsightsScraper failed: %s", e)
                company_profile = None

        return hot, tips, company_profile

    hot_skills, attitude_tips, company_profile = asyncio.run(_run_tools())

    SUGGESTIONS_CACHE[cache_key] = {
        "hot_skills": hot_skills,
        "attitude_tips": attitude_tips,
        "company": company_profile,
    }

    payload = {
        "role": role_title,
        "hot_skills": hot_skills,
        "attitude_tips": attitude_tips,
        "company": company_profile,
    }
    return jsonify(payload)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if not FRONTEND_DIST.exists():
        return (
            "Frontend build missing. Run `npm run build` inside `frontend/` first.",
            404,
        )

    target = FRONTEND_DIST / path
    if path and target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


asyncio.run(init())



if __name__ == "__main__":
    app.run(debug=True, port=5001)




