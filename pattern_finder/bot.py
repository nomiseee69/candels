"""Telegram-бот: меню паттернов, приём скриншотов, проверка распознавания, сканирование.

Долгие задачи (сканирование, история, разметка, автоскан) идут в фоновых потоках,
поэтому бот отвечает на кнопки, пока ищет паттерны."""

import itertools
import os
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

from catalog import ALL_PATTERNS, RULE_PATTERNS, SEARCHABLE
from notifier import (
    API, TOKEN, answer_callback, edit_keyboard, send_document, send_message, send_photo,
)
from autoscan import Schedule, run_autoscan
from pine import pine_help, pine_script
from market import top_symbols
from scanner import (
    ALL_INTERVALS, COIN_OPTIONS, MAX_CARDS, cards_overview, draw_coin, full_markup,
    group_by_coin, history_user, scan_user,
)
from storage import (
    add_pattern, delete_pattern, get_autoscan, get_coins, get_intervals, get_patterns,
    get_settings, set_coins, toggle_autoscan, toggle_interval, toggle_setting,
)
from texts import plural
from version import VERSION
from trendlines import KIND_NAMES, kind_from_drawing
from vision import analyze_screenshot, describe_lines, draw_debug

PATTERNS_DIR = "patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

waiting = {}       # chat_id → название паттерна, для которого ждём фото
MAX_MESSAGE = 4000
KEEP_CARDS = 300   # сколько последних находок помним на пользователя (для кнопок под ними)
OVERVIEW_BUTTONS = 39

# --- фоновые задачи ------------------------------------------------------------------------
jobs = ThreadPoolExecutor(max_workers=3)          # сканирования и история пользователей
autoscan_jobs = ThreadPoolExecutor(max_workers=1) # автоскан — по одному таймфрейму за раз
busy = set()                                      # кто сейчас ждёт результат
busy_lock = threading.Lock()
chart_lock = threading.Lock()                     # matplotlib не умеет рисовать в двух потоках сразу


def start_job(chat_id, func, *args):
    """Запускает долгую задачу в фоне. Одному человеку — одна задача за раз."""
    with busy_lock:
        if chat_id in busy:
            send_message("⏳ Ещё работаю над прошлым запросом — пришлю, как закончу.", chat_id)
            return
        busy.add(chat_id)

    def run():
        try:
            func(chat_id, *args)
        except Exception as error:
            print("Ошибка в фоновой задаче:", error)
            traceback.print_exc()
            send_message("😕 Что-то пошло не так, попробуй ещё раз чуть позже.", chat_id)
        finally:
            with busy_lock:
                busy.discard(chat_id)

    jobs.submit(run)


# --- находки: храним по номеру, чтобы кнопки под старыми сообщениями не путались -------------
card_ids = itertools.count(1)
cards_by_chat = {}       # chat_id → OrderedDict(номер → карточка)
cards_lock = threading.Lock()


def remember_cards(chat_id, cards):
    """Даёт каждой находке номер и запоминает. Возвращает номера."""
    with cards_lock:
        store = cards_by_chat.setdefault(chat_id, OrderedDict())
        ids = []
        for card in cards:
            card_id = next(card_ids)
            store[card_id] = card
            ids.append(card_id)
        while len(store) > KEEP_CARDS:
            store.popitem(last=False)
    return ids


def find_card(chat_id, card_id):
    with cards_lock:
        return cards_by_chat.get(chat_id, {}).get(card_id)

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
        {"text": "⚙️ Настройки", "callback_data": "settings"},
    ])
    auto_on = chat_id is not None and get_autoscan(chat_id)
    rows.append([{"text": "🔔 Автоскан: включён" if auto_on else "🔕 Автоскан: выключен",
                  "callback_data": "auto"}])
    return {"inline_keyboard": rows}


def show_menu(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    send_message(f"Выбери паттерн, который хочешь искать (таймфреймы: {tfs}, "
                 f"монеты: топ-{get_coins(chat_id)} по объёму):", chat_id, menu_keyboard(chat_id))


RULE_HELP = {
    "FVG": "FVG ищу по правилам на свечах: дыра между 1-й и 3-й свечой "
           "(значимые — от импульсной свечи, см. ⚙️ Настройки).",
    "BOS": "BOS ищу по правилам: цена ЗАКРЫЛАСЬ за последним свингом по тренду.",
    "CHoCH": "CHoCH ищу по правилам: первый слом структуры против тренда.",
    "EQH / EQL": "EQH/EQL ищу по правилам: 2+ свинга почти на одной цене, ликвидность ещё не снята.",
    "Order Block": "Order Block ищу по правилам: последняя свеча против движения перед сломом структуры. "
                   "Пришлю, когда блок сложился или цена вернулась к нему.",
    "Breaker Block": "Breaker ищу по правилам: Order Block, который пробили, — бывшее сопротивление "
                     "стало поддержкой (или наоборот).",
    "Mitigation Block": "Mitigation ищу по правилам: провал движения без снятия ликвидности "
                        "(более высокий лой / более низкий хай) и пробой свинга.",
}


def settings_text(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    return (f"⚙️ Настройки поиска\n"
            f"Таймфреймы: {tfs}\n"
            f"Монеты: топ-{get_coins(chat_id)} по объёму\n\n"
            "🎯 Значимые FVG — импульсная свеча сломала структуру, дыра ≥ 0.75 ATR и объём выше "
            "обычного; зона отработана, когда свеча закрылась за её серединой. "
            "«Все» — любая дыра больше 0.5 ATR.\n"
            "🧩 Только комбинации — присылать монеты, где одновременно нашлось 2+ твоих паттерна.")


def settings_keyboard(chat_id):
    settings = get_settings(chat_id)
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    fvg = "🎯 FVG: только значимые" if settings["fvg"] == "significant" else "🎯 FVG: все"
    combos = "🧩 Только комбинации: да" if settings["combos_only"] else "🧩 Только комбинации: нет"
    return {"inline_keyboard": [
        [{"text": f"⏱ Таймфреймы: {tfs}", "callback_data": "tf"}],
        [{"text": f"🪙 Монеты: топ-{get_coins(chat_id)}", "callback_data": "coins"}],
        [{"text": fvg, "callback_data": "set:fvg"}],
        [{"text": combos, "callback_data": "set:combos_only"}],
        [{"text": "⬅️ В меню", "callback_data": "menu"}],
    ]}


def coins_keyboard(chat_id):
    chosen = get_coins(chat_id)
    row = [
        {"text": ("✅ " if n == chosen else "") + str(n), "callback_data": f"coins:{n}"}
        for n in COIN_OPTIONS
    ]
    return {"inline_keyboard": [row]}


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

    if name in RULE_PATTERNS:
        # SMC по правилам: скрин — пример для тебя, искать его форму не нужно
        pattern_id = add_pattern(chat_id, name, path)
        keyboard = {"inline_keyboard": [[
            {"text": "✅ Ок", "callback_data": f"ok:{pattern_id}"},
            {"text": "❌ Удалить", "callback_data": f"bad:{pattern_id}"},
        ]]}
        send_message(f"Сохранил «{name}» 👍\n{RULE_HELP[name]}\n"
                     "Скрин оставил как твой пример — на рынке ищу по правилам, а не по картинке.",
                     chat_id, keyboard)
        return

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


CAPTION_LIMIT = 1000   # подпись к фото в Telegram — до 1024 символов


def send_card(chat_id, card_id, coin):
    """Одна монета со всеми таймфреймами: график + подпись + кнопка разметки на все ТФ."""
    path = f"card_{chat_id}_{card_id}.png"
    with chart_lock:
        draw_coin(coin, path)
    tfs = len({f["interval"] for f in coin["frames"]})
    keyboard = {"inline_keyboard": [[
        {"text": "📈 Разметка для TradingView" + (" (все ТФ)" if tfs > 1 else ""),
         "callback_data": f"pine:{card_id}"},
    ]]}
    caption = coin["caption"]
    try:
        if len(caption) <= CAPTION_LIMIT:
            send_photo(path, caption, chat_id, keyboard)
        else:                                  # длинная подпись — фото с заголовком, текст следом
            send_photo(path, caption.split("\n")[0] + "\n(подробности ниже)", chat_id)
            send_message(caption, chat_id, keyboard)
    finally:
        remove_file(path)


def overview_keyboard(ids, coins):
    buttons = [
        {"text": coin["symbol"].removesuffix("USDT"), "callback_data": f"card:{card_id}"}
        for card_id, coin in zip(ids[:OVERVIEW_BUTTONS], coins[:OVERVIEW_BUTTONS])
    ]
    return {"inline_keyboard": [buttons[i:i + 4] for i in range(0, len(buttons), 4)]}


def send_cards(chat_id, coins):
    """Лучшие монеты — графиками (все ТФ монеты в одном сообщении), остальные — списком
    с кнопками. coins — результат group_by_coin."""
    ids = remember_cards(chat_id, coins)
    for card_id, coin in zip(ids[:MAX_CARDS], coins[:MAX_CARDS]):
        send_card(chat_id, card_id, coin)
    rest = coins[MAX_CARDS:]
    if rest:
        text = cards_overview(rest)
        for part in split_text(text)[:-1]:
            send_message(part, chat_id)
        send_message(split_text(text)[-1], chat_id, overview_keyboard(ids[MAX_CARDS:], rest))


def show_card(chat_id, card_id):
    card = find_card(chat_id, card_id)
    if card is None:
        send_message("Эта находка устарела — запусти сканирование ещё раз.", chat_id)
        return
    send_card(chat_id, card_id, card)


def send_pine(chat_id, card_id):
    coin = find_card(chat_id, card_id)
    if coin is None:
        send_message("Эта находка устарела — запусти сканирование ещё раз.", chat_id)
        return
    intervals = get_intervals(chat_id)
    tfs = ", ".join(i.upper() for i in intervals)
    send_message(f"📈 Собираю разметку {coin['symbol']} по всем твоим таймфреймам ({tfs})…", chat_id)
    groups = full_markup(get_patterns(chat_id), coin, intervals, get_settings(chat_id))
    first = coin["frames"][0]["interval"]
    code = pine_script(coin["symbol"], first, groups)
    filename = f"{coin['symbol']}_patterns.pine.txt"
    path = f"pine_{chat_id}_{card_id}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    help_text = pine_help(coin["symbol"], first, groups)
    try:
        send_document(path, help_text, chat_id, filename)
    finally:
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
    coins = get_coins(chat_id)
    send_message(f"🔍 Сканирую топ-{coins} монет ({tfs}), {wait_text(coins, intervals)}…", chat_id)
    settings = get_settings(chat_id)
    cards, skipped = scan_user(patterns, intervals, symbols=top_symbols(coins), settings=settings)
    report_skipped(chat_id, skipped)
    if not cards:
        text = "Совпадений сейчас нет."
        if settings["combos_only"]:
            text += " Включено «только комбинации» — одиночные находки не показываю (⚙️ Настройки)."
        send_message(text, chat_id)
        return
    coins_found = group_by_coin(cards)
    combos = sum(any(len(f["items"]) > 1 for f in c["frames"]) for c in coins_found)
    multi = sum(len(c["frames"]) > 1 for c in coins_found)
    summary = f"Нашлось на {plural(len(coins_found), 'монете', 'монетах', 'монетах')}"
    extras = []
    if combos:
        extras.append(f"комбинации: {combos}")
    if multi:
        extras.append(f"на нескольких ТФ сразу: {multi}")
    if extras:
        summary += " (" + ", ".join(extras) + ")"
    more = f". Показываю {MAX_CARDS} лучших, остальные — списком." if len(coins_found) > MAX_CARDS else "."
    send_message(summary + more, chat_id)
    send_cards(chat_id, coins_found)


def wait_text(coins, intervals):
    """Примерно сколько ждать: ~0.15 с на монету и таймфрейм."""
    seconds = coins * len(intervals) * 0.15
    return "это займёт до минуты" if seconds <= 60 else f"это займёт ~{round(seconds / 60)} мин"


def run_history(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("Сначала добавь хотя бы один паттерн 👆", chat_id)
        return
    intervals = get_intervals(chat_id)
    coins = get_coins(chat_id)
    send_message(f"📊 Считаю историю по топ-{coins} монет, {wait_text(coins * 2, intervals)}…", chat_id)
    reports, cards, skipped = history_user(patterns, intervals, symbols=top_symbols(coins),
                                           settings=get_settings(chat_id))
    report_skipped(chat_id, skipped)
    for report in reports:
        for part in split_text(report):
            send_message(part, chat_id)
    send_cards(chat_id, group_by_coin(cards))


def handle_callback(query):
    chat_id = query["message"]["chat"]["id"]
    data = query["data"]
    answer_callback(query["id"])

    if data == "my":
        show_my_patterns(chat_id)
        return

    if data == "scan":
        start_job(chat_id, run_scan)
        return

    if data == "history":
        start_job(chat_id, run_history)
        return

    if data == "coins":
        send_message("Сколько самых торгуемых монет Binance проверять? Чем больше — тем дольше "
                     "сканирование, но больше находок.", chat_id, coins_keyboard(chat_id))
        return

    if data == "settings":
        send_message(settings_text(chat_id), chat_id, settings_keyboard(chat_id))
        return

    if data == "menu":
        show_menu(chat_id)
        return

    if data == "tf":
        send_message("На каких таймфреймах искать? Нажимай, чтобы включить или выключить:",
                     chat_id, interval_keyboard(chat_id))
        return

    if data == "tf_done":
        send_message(settings_text(chat_id), chat_id, settings_keyboard(chat_id))
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
        start_job(chat_id, send_pine, int(value))
        return

    if action == "card":
        start_job(chat_id, show_card, int(value))
        return

    if action == "coins":
        set_coins(chat_id, int(value))
        send_message(f"🪙 Буду проверять топ-{value} монет по объёму торгов.", chat_id)
        send_message(settings_text(chat_id), chat_id, settings_keyboard(chat_id))
        return

    if action == "set":
        settings = toggle_setting(chat_id, value)
        edit_keyboard(chat_id, query["message"]["message_id"], settings_keyboard(chat_id))
        searchable = {p["name"] for p in get_patterns(chat_id) if p["name"] in SEARCHABLE}
        if value == "combos_only" and settings["combos_only"] and len(searchable) < 2:
            send_message("⚠️ Для комбинаций нужно минимум 2 разных паттерна — сейчас находок не будет. "
                         "Добавь ещё паттерн из меню.", chat_id)
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
def autoscan_job(intervals):
    try:
        sent = run_autoscan(intervals, send_cards, send_message)
        print(f"Автоскан {', '.join(intervals)}: отправлено монет {sent}")
    except Exception as error:
        print(f"Ошибка автоскана {intervals}:", error)
        traceback.print_exc()


def check_schedule(schedule):
    """Если закрылись свечи (в 00:00 UTC закрываются сразу 1H, 4H, 12H и 1D) —
    один автоскан на все эти ТФ, чтобы монета пришла одним сообщением."""
    due = schedule.due(time.time())
    if due:
        autoscan_jobs.submit(autoscan_job, due)


def main():
    offset = 0
    schedule = Schedule(time.time())
    print(f"Бот запущен (обновление {VERSION}). Остановить: Ctrl+C")
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
