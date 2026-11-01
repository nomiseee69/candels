"""Отправка сообщений и картинок в Telegram."""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"

TV_INTERVALS = {"1h": "60", "4h": "240", "12h": "720", "1d": "D", "1w": "W"}


def _with_keyboard(data, keyboard):
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard)
    return data


def send_message(text, chat_id=CHAT_ID, keyboard=None):
    data = _with_keyboard({"chat_id": chat_id, "text": text}, keyboard)
    response = requests.post(f"{API}/sendMessage", data=data, timeout=10)
    return response.status_code == 200


def send_photo(path, caption, chat_id=CHAT_ID, keyboard=None):
    data = _with_keyboard({"chat_id": chat_id, "caption": caption}, keyboard)
    with open(path, "rb") as photo:
        response = requests.post(f"{API}/sendPhoto", data=data, files={"photo": photo}, timeout=30)
    return response.status_code == 200


def send_document(path, caption, chat_id=CHAT_ID, filename=None):
    """Файл, который пользователь скачает (например, скрипт для TradingView)."""
    with open(path, "rb") as f:
        response = requests.post(
            f"{API}/sendDocument",
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (filename or os.path.basename(path), f)},
            timeout=30,
        )
    return response.status_code == 200


def edit_keyboard(chat_id, message_id, keyboard):
    """Меняет кнопки под уже отправленным сообщением (галочки таймфреймов)."""
    requests.post(
        f"{API}/editMessageReplyMarkup",
        data={"chat_id": chat_id, "message_id": message_id, "reply_markup": json.dumps(keyboard)},
        timeout=10,
    )


def answer_callback(callback_id):
    requests.post(f"{API}/answerCallbackQuery", data={"callback_query_id": callback_id}, timeout=10)


def tradingview_link(symbol, interval):
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={TV_INTERVALS[interval]}"


if __name__ == "__main__":
    print(send_message("Привет! Бот на связи 🚀"))
