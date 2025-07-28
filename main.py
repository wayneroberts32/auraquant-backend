from fastapi import FastAPI
import os
import requests

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AuraQuant backend is running."}

@app.get("/test/ping-telegram")
def ping_telegram():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return {"error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"}

    message = "📡 Rich_Bot Ping: Live and working via Render backend!"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)

    return {
        "status_code": response.status_code,
        "response": response.text
    }
