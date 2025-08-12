# Webhook_Telegram.py
# ===================
# AuraQuant Rich_Bot Telegram Webhook (standalone)
# Uses env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

from fastapi import FastAPI, Request
import httpx
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Webhook_Telegram")

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

@app.post("/")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {"raw": (await request.body()).decode("utf-8", errors="ignore")}

    logger.info(f"[WEBHOOK RECEIVED] {payload}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
        return {"status": "error", "message": "Telegram env vars not set"}

    message = (
        f"📡 <b>Rich_Bot Webhook Triggered</b>\n"
        f"<b>Time:</b> {datetime.utcnow()} UTC\n"
        f"<b>Payload:</b> {payload}"
    )

    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                telegram_url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            )
        logger.info(f"Telegram send status: {r.status_code} {r.text[:200]}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error sending to Telegram: {e}")
        return {"status": "error", "details": str(e)}
