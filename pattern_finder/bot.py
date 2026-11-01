"""Telegram-бот: меню паттернов, приём скриншотов, проверка распознавания, сканирование."""

import os
import time
import traceback

import requests

from catalog import ALL_PATTERNS, SEARCHABLE
from notifier import (
    API, TOKEN, answer_callback, edit_keyboard, send_document, send_message, send_photo,
)
from autoscan import Schedule, run_autoscan
from pine import PINE_HELP, pine_script
from scanner import ALL_INTERVALS, draw_card, full_markup, history_user, scan_user
from storage import (
    add_pattern, delete_pattern, get_autoscan, get_intervals, get_patterns,
    toggle_autoscan, toggle_interval,
)
from trendlines import KIND_NAMES, kind_from_drawing
from vision import analyze_screenshot, describe_lines, draw_debug

PATTERNS_DIR = "patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

waiting = {}      # chat_id → название паттерна, для которого ждём фото
last_cards = {}   # chat_id → карточки последнего сканирования/истории (для кнопки разметки)
MAX_MESSAGE = 4000

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
def menu_keyboard(chat_id=None):
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
    rows.append([
        {"text": "📊 История", "callback_data": "history"},
        {"text": "⏱ Таймфреймы", "callback_data": "tf"},
    ])
    auto_on = chat_id is not None and get_autoscan(chat_id)
    rows.append([{"text": "🔔 Автоскан: включён" if auto_on else "🔕 Автоскан: выключен",
                  "callback_data": "auto"}])
    return {"inline_keyboard": rows}


def show_menu(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    send_message(f"Выбери паттерн, который хочешь искать (таймфреймы: {tfs}):", chat_id, menu_keyboard(chat_id))


def interval_keyboard(chat_id):
    chosen = get_intervals(chat_id)
    row = [
        {"text": ("✅ " if i in chosen else "▫️ ") + i.upper(), "callback_data": f"tf:{i}"}
        for i in ALL_INTERVALS
    ]
    return {"inline_keyboard": [row, [{"text": "Готово", "callback_data": "tf_done"}]]}


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


def split_text(text, limit=MAX_MESSAGE):
    """Telegram принимает до 4096 символов — длинный отчёт режем по абзацам."""
    parts, current = [], ""
    for block in text.split("\n\n"):
        if current and len(current) + len(block) + 2 > limit:
            parts.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        parts.append(current)
    return parts


def send_cards(chat_id, cards):
    """Отправляет карточки: график + подпись + кнопка «разметка для TradingView»."""
    last_cards[chat_id] = cards
    for index, card in enumerate(cards):
        path = draw_card(card, f"card_{chat_id}_{index}.png")
        keyboard = {"inline_keyboard": [[
            {"text": "📈 Разметка для TradingView", "callback_data": f"pine:{index}"},
        ]]}
        send_photo(path, card["caption"], chat_id, keyboard)
        remove_file(path)


def send_pine(chat_id, index):
    cards = last_cards.get(chat_id, [])
    if index >= len(cards):
        send_message("Эта находка устарела — запусти сканирование ещё раз.", chat_id)
        return
    card = cards[index]
    send_message("📈 Собираю разметку всех паттернов на этом графике…", chat_id)
    groups = full_markup(get_patterns(chat_id), card)
    code = pine_script(card["symbol"], card["interval"], groups)
    filename = f"{card['symbol']}_{card['interval']}_patterns.pine.txt"
    path = f"pine_{chat_id}_{index}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    help_text = PINE_HELP.format(symbol=card["symbol"], interval=card["interval"])
    send_document(path, help_text, chat_id, filename)
    remove_file(path)


def report_skipped(chat_id, skipped):
    if skipped:
        send_message("Пока не умею искать: " + ", ".join(skipped) + ". Детекторы скоро появятся.", chat_id)


def run_scan(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("Сначала добавь хотя бы один паттерн 👆", chat_id)
        return
    intervals = get_intervals(chat_id)
    tfs = ", ".join(i.upper() for i in intervals)
    send_message(f"🔍 Сканирую рынок ({tfs}), это займёт до минуты…", chat_id)
    cards, skipped = scan_user(patterns, intervals)
    report_skipped(chat_id, skipped)
    if not cards:
        send_message("Совпадений сейчас нет.", chat_id)
    else:
        send_cards(chat_id, cards)


def run_history(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("Сначала добавь хотя бы один паттерн 👆", chat_id)
        return
    intervals = get_intervals(chat_id)
    send_message("📊 Считаю историю по всем монетам, это займёт до минуты…", chat_id)
    reports, cards, skipped = history_user(patterns, intervals)
    report_skipped(chat_id, skipped)
    for report in reports:
        for part in split_text(report):
            send_message(part, chat_id)
    send_cards(chat_id, cards)


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

    if data == "history":
        run_history(chat_id)
        return

    if data == "tf":
        send_message("На каких таймфреймах искать? Нажимай, чтобы включить или выключить:",
                     chat_id, interval_keyboard(chat_id))
        return

    if data == "tf_done":
        show_menu(chat_id)
        return

    if data == "auto":
        on = toggle_autoscan(chat_id)
        tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
        if on:
            send_message(f"🔔 Автоскан включён ({tfs}). После закрытия каждой свечи проверю рынок "
                         "и пришлю только новые находки.", chat_id)
        else:
            send_message("🔕 Автоскан выключен.", chat_id)
        show_menu(chat_id)
        return

    action, _, value = data.partition(":")

    if action == "tf":
        toggle_interval(chat_id, value, ALL_INTERVALS)
        edit_keyboard(chat_id, query["message"]["message_id"], interval_keyboard(chat_id))
        return

    if action == "pine":
        send_pine(chat_id, int(value))
        return

    if action == "pick":
        name = ALL_PATTERNS[int(value)]
        waiting[chat_id] = name
        send_message(f"Пришли скриншот паттерна «{name}» 📷\n\n{TIPS}", chat_id)
        return

    if action == "ok":
        send_message("✅ Сохранил. Добавь ещё паттерн или нажми «Сканировать».", chat_id, menu_keyboard(chat_id))
        return

    if action in ("bad", "del"):
        removed = delete_pattern(chat_id, value)
        if removed is None:
            send_message("Этот паттерн уже удалён.", chat_id)
            return
        remove_file(removed["photo"])
        send_message(f"🗑 Удалил «{removed['name']}»", chat_id)
        if action == "bad":
            send_message(TIPS, chat_id, menu_keyboard(chat_id))


# --- главный цикл --------------------------------------------------------------------------
def check_schedule(schedule):
    """Если у какого-то таймфрейма закрылась свеча — автоскан для подписчиков."""
    for interval in schedule.due(time.time()):
        try:
            sent = run_autoscan(interval, send_cards, send_message)
            print(f"Автоскан {interval}: отправлено карточек {sent}")
        except Exception as error:
            print(f"Ошибка автоскана {interval}:", error)
            traceback.print_exc()


def main():
    offset = 0
    schedule = Schedule(time.time())
    print("Бот запущен. Остановить: Ctrl+C")
    while True:
        check_schedule(schedule)
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
