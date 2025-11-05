import uvicorn
import time
import json
import os
import threading
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
import logging

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === FastAPI app ===
app = FastAPI(title="Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Path log JSON ===
LOG_FILE = "/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source/API/API_Response_Logs/translation_log.json"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# === Hàm ghi log JSON nền ===
def save_translation_log(text, translated_text, response_time, timestamp):
    log_entry = {
        "timestamp": timestamp,
        "text": text,
        "translated_text": translated_text,
        "response_time": round(response_time, 4)
    }

    log_data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    log_data = data
        except Exception:
            log_data = []

    # Giữ 7 ngày
    seven_days_ago = datetime.now() - timedelta(days=7)
    filtered_log_data = []
    for entry in log_data:
        try:
            entry_time = datetime.strptime(entry.get("timestamp", ""), "%Y-%m-%dT%H:%M:%S")
            if entry_time >= seven_days_ago:
                filtered_log_data.append(entry)
        except Exception:
            filtered_log_data.append(entry)

    filtered_log_data.append(log_entry)

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_log_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write log: {e}")

# === Endpoint translate (trả kết quả trước, log sau) ===
@app.get("/translate")
async def translate(text: str = Query(..., description="Text to translate from Vietnamese to English")):
    start_time = time.time()
    try:
        # Dịch trước
        translated = GoogleTranslator(source='vi', target='en').translate(text)
        response_time = time.time() - start_time

        # Trả kết quả ngay
        result = {"success": True, "translated_text": translated}

        # Ghi log nền
        threading.Thread(
            target=save_translation_log,
            args=(text, translated, response_time, datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
            daemon=True
        ).start()

        # Log console nền
        threading.Thread(
            target=lambda: logger.info(f"Text: {text} | Translated: {translated} | Response_time: {response_time:.4f}s"),
            daemon=True
        ).start()

        return result
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Run ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8598)
