# Webhook_Telegram.py
# AuraQuant Rich_Bot Telegram Webhook (standalone)
# Sends incoming webhook payloads to a Telegram chat

from fastapi import FastAPI, Request
import httpx
import logging
from datetime import datetime

# ==============================
# 🔹 HARDCODED TELEGRAM SETTINGS
# ==============================
# Replace with your bot's API token and chat ID for testing
TELEGRAM_BOT_TOKEN = 8194171444:AAEqUFMLyhlIIQLHSyf1y-9og8YsRQubfiY
TELEGRAM_CHAT_ID = 6995384125
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Webhook_Telegram")

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "webhook-alive"}

# Handle both `/webhook` and `/webhook/`
@app.post("/webhook")
@app.post("/webhook/")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {"raw": (await request.body()).decode("utf-8", errors="ignore")}

    logger.info(f"[WEBHOOK RECEIVED] {payload}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing Telegram bot token or chat ID")
        return {"status": "error", "message": "Telegram credentials not set"}

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
