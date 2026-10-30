import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"

TV_INTERVALS = {"1h":"60", "4h": "240", "12h": "720", "1d": "D", "1w": "W"}




def send_message(text, chat_id=CHAT_ID, keyboard=None):
    data = {"chat_id": chat_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard)
    response = requests.post(f"{API}/sendMessage", data=data, timeout=10)
    return response.status_code == 200

def send_photo(path, caption, chat_id=CHAT_ID):
    with open(path, "rb") as photo:
        response = requests.post(
            f"{API}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": photo},
            timeout=30,
        )
    return response.status_code == 200

def answer_callback(callback_id):
    requests.post(f"{API}/answerCallbackQuery", data={"callback_query_id": callback_id}, timeout=10)


def tradingview_link(symbol, interval):
    tv_interval = TV_INTERVALS[interval]
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={tv_interval}"


if __name__ == "__main__":
    print(send_message("Привет! Бот на связи 🚀"))