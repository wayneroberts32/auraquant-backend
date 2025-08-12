# Webhook_Telegram.py
# ===================
# AuraQuant Rich_Bot Telegram Webhook
# Matches DivineConfig style + Render logging
# ===========================================

from fastapi import FastAPI, Request
import httpx
import logging
from datetime import datetime

# Import DivineConfig from main.py
try:
    from main import DivineConfig
except ImportError:
    from DivineConfig import DivineConfig  # fallback if separated later

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Webhook_Telegram")

app = FastAPI()

@app.post("/")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
        logger.info(f"[WEBHOOK RECEIVED] {payload}")

        # Format message
        message = (
            f"📡 <b>Rich_Bot Webhook Triggered</b>\n"
            f"<b>Time:</b> {datetime.utcnow()} UTC\n"
            f"<b>Payload:</b> {payload}"
        )

        # Send to Telegram
        telegram_url = f"https://api.telegram.org/bot{DivineConfig.TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(telegram_url, json={
                "chat_id": DivineConfig.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            })

        return {"status": "ok", "message": "Alert sent to Telegram"}

    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        return {"status": "error", "details": str(e)}

