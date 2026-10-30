import time

import requests

from notifier import API, TOKEN, send_message
from scanner import scan_market, send_results


def get_updates(offset):
    response = requests.get(
        f"{API}/getUpdates",
        params={"offset": offset, "timeout": 30},
        timeout=40,
    )
    return response.json()["result"]


def download_photo(file_id, path):
    info = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=10).json()
    file_path = info["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    content = requests.get(url, timeout=30).content
    with open(path, "wb") as f:
        f.write(content)


def handle_message(message):
    chat_id = message["chat"]["id"]

    if "photo" not in message:
        send_message(
            "Пришли скриншот паттерна: тёмная линия на светлом фоне, без шкал и подписей.",
            chat_id,
        )
        return

    file_id = message["photo"][-1]["file_id"]
    path = f"user_{chat_id}.jpg"
    download_photo(file_id, path)
    send_message("Фото получил, ищу похожие паттерны на 10 монетах…", chat_id)

    good = scan_market(path)
    if not good:
        send_message("Свежих совпадений нет. Попробуй позже или пришли другой паттерн.", chat_id)
    else:
        send_results(good, chat_id)


def main():
    offset = 0
    print("Бот запущен. Остановить: Ctrl+C")
    while True:
        try:
            updates = get_updates(offset)
        except requests.RequestException as error:
            print("Ошибка сети:", error)
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message:
                continue
            try:
                handle_message(message)
            except Exception as error:
                print("Ошибка при обработке:", error)
                send_message("Не получилось обработать сообщение 😕", message["chat"]["id"])


if __name__ == "__main__":
    main()