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


def _reply(data, reply_to):
    """Ответ на сообщение: в Telegram над новым сообщением видна цитата, нажал — перешёл к нему."""
    if reply_to is not None:
        data["reply_parameters"] = json.dumps({"message_id": reply_to, "allow_sending_without_reply": True})
    return data


def _message_id(response):
    """Номер отправленного сообщения (чтобы потом его менять) или None, если не отправилось."""
    if response.status_code != 200:
        return None
    return response.json().get("result", {}).get("message_id")


def send_message(text, chat_id=CHAT_ID, keyboard=None, reply_to=None):
    """Отправляет текст. Возвращает номер сообщения или None."""
    data = _reply(_with_keyboard({"chat_id": chat_id, "text": text}, keyboard), reply_to)
    response = requests.post(f"{API}/sendMessage", data=data, timeout=10)
    return _message_id(response)


def edit_photo(chat_id, message_id, path, caption, keyboard=None):
    """Меняет картинку, подпись и кнопки сообщения-просмотрщика на месте (листание монет и примеров)."""
    media = {"type": "photo", "media": "attach://photo", "caption": caption}
    data = _with_keyboard({"chat_id": chat_id, "message_id": message_id, "media": json.dumps(media)}, keyboard)
    with open(path, "rb") as photo:
        response = requests.post(f"{API}/editMessageMedia", data=data, files={"photo": photo}, timeout=30)
    return response.status_code == 200


def delete_message(chat_id, message_id):
    requests.post(f"{API}/deleteMessage", data={"chat_id": chat_id, "message_id": message_id}, timeout=10)


def edit_message(chat_id, message_id, text, keyboard=None):
    """Меняет текст и кнопки уже отправленного сообщения — меню листается на месте,
    без новых сообщений в чате."""
    data = _with_keyboard({"chat_id": chat_id, "message_id": message_id, "text": text}, keyboard)
    response = requests.post(f"{API}/editMessageText", data=data, timeout=10)
    return response.status_code == 200


def send_action(chat_id, action="upload_document"):
    """Статус «отправляет файл…» вверху чата, пока бот готовит ответ."""
    requests.post(f"{API}/sendChatAction", data={"chat_id": chat_id, "action": action}, timeout=10)


def send_photo(path, caption, chat_id=CHAT_ID, keyboard=None):
    """Отправляет картинку. Возвращает номер сообщения или None."""
    data = _with_keyboard({"chat_id": chat_id, "caption": caption}, keyboard)
    with open(path, "rb") as photo:
        response = requests.post(f"{API}/sendPhoto", data=data, files={"photo": photo}, timeout=30)
    return _message_id(response)


def send_document(path, caption, chat_id=CHAT_ID, filename=None, keyboard=None, reply_to=None):
    """Файл, который пользователь скачает (например, скрипт для TradingView)."""
    data = _reply(_with_keyboard({"chat_id": chat_id, "caption": caption}, keyboard), reply_to)
    with open(path, "rb") as f:
        response = requests.post(
            f"{API}/sendDocument",
            data=data,
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
