from __future__ import annotations
from typing import Any

import httpx
from mcp.server import FastMCP
import os , logging
mcp =FastMCP('Chatbot')

logging.basicConfig(level=logging.INFO)


WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
WHATSAPP_PHONE_ID = os.environ["WHATSAPP_PHONE_ID"]
WHATSAPP_API_BASE = "https://graph.facebook.com/v21.0"

async def send_report(body):
    url =  f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_ID}/messages"
    to = "owner"
    headers  = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp =await client.post(url, headers= headers, json = payload)
        resp.raise_for_status()
        return resp.json()




@mcp.tool()
async def notify_user(user_phone: str,
    message: str):
    logging.info("Sending WhatsApp message to %s", user_phone)
    return await send_report(body=message)


if __name__ == "__main__":
    mcp.run()