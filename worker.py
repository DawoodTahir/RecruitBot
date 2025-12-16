import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from utils import GraphRAG, Preprocess


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot-worker")


AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
S3_BUCKET_RESUMES = os.environ.get("S3_BUCKET_RESUMES", "recruitlens-resumes")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")


def _make_aws_clients():
    session = boto3.session.Session()
    common = {"region_name": AWS_REGION}
    if AWS_ENDPOINT_URL:
        common["endpoint_url"] = AWS_ENDPOINT_URL
    s3 = session.client("s3", **common)
    sqs = session.client("sqs", **common)
    return s3, sqs


async def _index_resume_from_s3(
    rag: GraphRAG,
    preprocessor: Preprocess,
    s3,
    user_id: str,
    s3_key: str,
    ext: str,
) -> None:
    """
    Download a resume object from S3/LocalStack, normalise it to text using the
    same rules as /upload, and index it into GraphRAG using the given user_id.
    """
    logger.info("Worker: processing resume for user=%s key=%s", user_id, s3_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        local_path = tmp_dir / Path(s3_key).name

        try:
            s3.download_file(S3_BUCKET_RESUMES, s3_key, str(local_path))
        except (BotoCoreError, ClientError) as exc:
            logger.error("Worker: failed to download %s from S3: %s", s3_key, exc)
            return

        index_path = local_path
        ext_lower = (ext or "").lower()
        try:
            if ext_lower == ".pdf":
                txt_path = local_path.with_name(f"{local_path.name}.txt")
                preprocessor.extract_pdf(str(local_path), str(txt_path))
                index_path = txt_path
            elif ext_lower in (".docx", ".doc"):
                txt_path = local_path.with_name(f"{local_path.name}.txt")
                preprocessor.extract_docx_to_text(str(local_path), str(txt_path))
                index_path = txt_path
        except Exception as exc:
            logger.error("Worker: failed to extract text for %s: %s", local_path, exc)
            return

        async def _run():
            await rag.index_document(str(index_path), user_id=user_id)

        await _run()
        logger.info("Worker: successfully indexed resume for user=%s", user_id)


def _process_message(rag: GraphRAG, preprocessor: Preprocess, s3, message: dict) -> None:
    body_raw = message.get("Body", "")
    try:
        payload = json.loads(body_raw)
    except json.JSONDecodeError:
        logger.warning("Worker: received non-JSON message body: %r", body_raw)
        return

    if payload.get("type") != "resume_uploaded":
        logger.info("Worker: ignoring message with type=%r", payload.get("type"))
        return

    user_id = str(payload.get("user_id") or "anonymous")
    s3_key = payload.get("s3_key")
    ext = payload.get("ext") or ""

    if not s3_key:
        logger.warning("Worker: resume_uploaded message missing s3_key: %r", payload)
        return

    asyncio.run(_index_resume_from_s3(rag, preprocessor, s3, user_id, s3_key, ext))


def main() -> None:
    if not SQS_QUEUE_URL:
        logger.error("Worker: SQS_QUEUE_URL is not set; exiting.")
        return

    s3, sqs = _make_aws_clients()
    rag = GraphRAG()
    preprocessor = Preprocess()

    logger.info("Worker: started. Polling queue: %s", SQS_QUEUE_URL)

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=60,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.error("Worker: error receiving from SQS: %s", exc)
            continue

        messages = resp.get("Messages", [])
        if not messages:
            continue

        for msg in messages:
            receipt = msg.get("ReceiptHandle")
            try:
                _process_message(rag, preprocessor, s3, msg)
                if receipt:
                    sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Worker: error processing message: %s", exc)


if __name__ == "__main__":
    main()


