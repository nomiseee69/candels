"""Telegram-бот: меню паттернов, приём скриншотов, проверка распознавания, сканирование."""

import os
import time
import traceback

import requests

from catalog import ALL_PATTERNS, SEARCHABLE
from notifier import API, TOKEN, answer_callback, send_message, send_photo
from scanner import scan_user, send_scan_results
from storage import add_pattern, delete_pattern, get_patterns
from trendlines import KIND_NAMES, kind_from_drawing
from vision import analyze_screenshot, describe_lines, draw_debug

PATTERNS_DIR = "patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

waiting = {}   # chat_id → название паттерна, для которого ждём фото

TIPS = (
    "Как сделать скриншот, чтобы я его понял:\n"
    "• обрежь шкалу цены и времени\n"
    "• скрой индикаторы и объём\n"
    "• свечи или линия — подходит любая тема, светлая или тёмная\n"
    "• можешь нарисовать поверх линии или ломаную — я их увижу"
)


# --- Telegram API ----------------------------------------------------------------
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
    content = requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{file_path}", timeout=30).content
    with open(path, "wb") as f:
        f.write(content)


def remove_file(path):
    if path and os.path.exists(path):
        os.remove(path)


# --- меню ----------------------------------------------------------------------------
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
    rows.append([
        {"text": "📂 Мои паттерны", "callback_data": "my"},
        {"text": "🔍 Сканировать", "callback_data": "scan"},
    ])
    return {"inline_keyboard": rows}


def show_menu(chat_id):
    send_message("Выбери паттерн, который хочешь искать:", chat_id, menu_keyboard())


# --- приём фото ------------------------------------------------------------------------
def receive_pattern_photo(chat_id, name, file_id):
    path = f"{PATTERNS_DIR}/{chat_id}_{int(time.time() * 1000)}.jpg"
    download_photo(file_id, path)

    analysis = analyze_screenshot(path)
    if not analysis.ok:
        remove_file(path)
        send_message("Не смог разобрать график на картинке 😕\n\n" + TIPS, chat_id)
        show_menu(chat_id)
        return

    lines_text = describe_lines(analysis.lines)
    figure = kind_from_drawing(analysis.lines)
    pattern_id = add_pattern(chat_id, name, path, lines_text, figure)

    debug_path = f"{PATTERNS_DIR}/{chat_id}_debug.png"
    draw_debug(path, analysis, debug_path)
    chart_kind = "свечной график" if analysis.mode == "candles" else "линию цены"
    caption = f"Вот что я увидел в «{name}»:\n• {chart_kind} — жёлтым\n• {lines_text}\n"
    if figure:
        caption += f"• Фигура: {KIND_NAMES[figure]}\n"
    if name not in SEARCHABLE:
        caption += "\nИскать этот паттерн на рынке я пока не умею — детектор скоро появится.\n"
    caption += "\nВсё верно?"
    keyboard = {"inline_keyboard": [[
        {"text": "✅ Верно", "callback_data": f"ok:{pattern_id}"},
        {"text": "❌ Нет, удалить", "callback_data": f"bad:{pattern_id}"},
    ]]}
    send_photo(debug_path, caption, chat_id, keyboard)
    remove_file(debug_path)


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
    file_id = message["photo"][-1]["file_id"]
    receive_pattern_photo(chat_id, name, file_id)


# --- кнопки ------------------------------------------------------------------------------
def show_my_patterns(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("У тебя пока нет паттернов.", chat_id)
        return
    lines = [f"{i}. {p['name']}" for i, p in enumerate(patterns, start=1)]
    buttons = [
        [{"text": f"🗑 Удалить {i}", "callback_data": f"del:{p['id']}"}]
        for i, p in enumerate(patterns, start=1)
    ]
    send_message("Твои паттерны:\n" + "\n".join(lines), chat_id, {"inline_keyboard": buttons})


def run_scan(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("Сначала добавь хотя бы один паттерн 👆", chat_id)
        return
    send_message("🔍 Сканирую рынок, это займёт до минуты…", chat_id)
    findings, skipped = scan_user(patterns)
    if skipped:
        send_message("Пока не умею искать: " + ", ".join(skipped) + ". Детекторы скоро появятся.", chat_id)
    if not findings:
        send_message("Совпадений сейчас нет.", chat_id)
    else:
        send_scan_results(findings, chat_id)


def handle_callback(query):
    chat_id = query["message"]["chat"]["id"]
    data = query["data"]
    answer_callback(query["id"])

    if data == "my":
        show_my_patterns(chat_id)
        return

    if data == "scan":
        run_scan(chat_id)
        return

    action, _, value = data.partition(":")

    if action == "pick":
        name = ALL_PATTERNS[int(value)]
        waiting[chat_id] = name
        send_message(f"Пришли скриншот паттерна «{name}» 📷\n\n{TIPS}", chat_id)
        return

    if action == "ok":
        send_message("✅ Сохранил. Добавь ещё паттерн или нажми «Сканировать».", chat_id, menu_keyboard())
        return

    if action in ("bad", "del"):
        removed = delete_pattern(chat_id, value)
        if removed is None:
            send_message("Этот паттерн уже удалён.", chat_id)
            return
        remove_file(removed["photo"])
        send_message(f"🗑 Удалил «{removed['name']}»", chat_id)
        if action == "bad":
            send_message(TIPS, chat_id, menu_keyboard())


# --- главный цикл --------------------------------------------------------------------------
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
