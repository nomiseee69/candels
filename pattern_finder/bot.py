import os
import time
import traceback

import requests

from catalog import ALL_PATTERNS
from notifier import API, TOKEN, send_message, answer_callback
from storage import add_pattern, get_patterns, delete_pattern

waiting = {}
os.makedirs("patterns", exist_ok=True)

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
def menu_keyboard():
    rows = []
    row = []
    for i, name in enumerate(ALL_PATTERNS):
        row.append({"text": name, "callback_data": f"pick:{i}"})
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([{"text": "📂 Мои паттерны", "callback_data": "my"}])
    return {"inline_keyboard": rows}


def show_menu(chat_id):
    send_message("Выбери паттерн, который хочешь искать:", chat_id, menu_keyboard())


def handle_message(message):
    chat_id = message["chat"]["id"]

    if "photo" not in message:
        show_menu(chat_id)
        return

    if chat_id not in waiting:
        send_message("Сначала выбери, какой это паттерн 👇", chat_id)
        show_menu(chat_id)
        return

    name = waiting.pop(chat_id)
    number = len(get_patterns(chat_id)) + 1
    path = f"patterns/{chat_id}_{number}.jpg"
    file_id = message["photo"][-1]["file_id"]
    download_photo(file_id, path)
    add_pattern(chat_id, name, path)
    send_message(f"✅ Сохранил «{name}». Можешь добавить ещё:", chat_id)
    show_menu(chat_id)


def handle_callback(query):
    chat_id = query["message"]["chat"]["id"]
    data = query["data"]
    answer_callback(query["id"])

    if data == "my":
        patterns = get_patterns(chat_id)
        if not patterns:
            send_message("У тебя пока нет паттернов.", chat_id)
            return
        lines = [f"{i}. {p['name']}" for i, p in enumerate(patterns, start=1)]
        buttons = [
            [{"text": f"🗑 Удалить {i}", "callback_data": f"del:{i - 1}"}]
            for i in range(1, len(patterns) + 1)
        ]
        send_message("Твои паттерны:\n" + "\n".join(lines), chat_id, {"inline_keyboard": buttons})
        return

    if data.startswith("del:"):
        index = int(data.split(":")[1])
        removed = delete_pattern(chat_id, index)
        send_message(f"🗑 Удалил «{removed['name']}»", chat_id)
        return


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
            try:
                if "message" in update:
                    handle_message(update["message"])
                elif "callback_query" in update:
                    handle_callback(update["callback_query"])
            except Exception as error:
                print("Ошибка при обработке:", error)
                traceback.print_exc()

if __name__ == "__main__":
    main()