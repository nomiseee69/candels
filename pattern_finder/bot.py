"""Telegram-бот: меню паттернов, приём скриншотов, проверка распознавания, сканирование.

Долгие задачи (сканирование, история, разметка, автоскан) идут в фоновых потоках,
поэтому бот отвечает на кнопки, пока ищет паттерны."""

import itertools
import re
import os
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

from catalog import ALL_PATTERNS, RULE_PATTERNS, SEARCHABLE, SHAPE_PATTERNS, SMC_PATTERNS, TA_PATTERNS
from notifier import (
    API, TOKEN, TV_INTERVALS, answer_callback, delete_message, edit_keyboard, edit_message, edit_photo,
    send_action, send_document, send_message, send_photo, set_commands, tradingview_link,
)
from autoscan import Schedule, run_autoscan
from pine import pine_help, pine_script
from market import top_symbols
from scanner import (
    ALL_INTERVALS, CANDLES, COIN_OPTIONS, build_history_jobs, draw_coin, full_markup,
    group_by_coin, history_user, scan_user, selected_markup,
)
from texts import quality_text
from storage import (
    add_feedback, add_pattern, clear_feedback, delete_pattern, first_visit, get_autoscan, get_coins,
    get_feedback, get_intervals, get_patterns, get_settings, set_coins, toggle_autoscan,
    toggle_interval, toggle_setting,
)
from chart import plot_example, plot_findings
from guide import COMMANDS, DEMO_LEGEND, PATTERN_INFO, WELCOME
from templates import template_shape
from feedback import MIN_VOTES_FIT, features, learn, make_judge, rule_text
from feedback import summary as learning_summary
from similarity import example_from_item
from fingerprint import FAMILY, describe as describe_fingerprint, kind_text, make_fingerprint, screenshot_fingerprint
from vision import extract_candles
from confluence import direction, score as confluence_score
from market import candles_to_df, fetch_many, get_candles
from texts import plural
from version import VERSION
from trendlines import KIND_NAMES, kind_from_drawing
from vision import analyze_screenshot, describe_lines, draw_debug

PATTERNS_DIR = "patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

waiting = {}       # chat_id → название паттерна, для которого ждём фото
MAX_MESSAGE = 4000
KEEP_CARDS = 300   # сколько последних находок помним на пользователя (для кнопок под ними)

# --- фоновые задачи ------------------------------------------------------------------------
jobs = ThreadPoolExecutor(max_workers=3)          # сканирования и история пользователей
autoscan_jobs = ThreadPoolExecutor(max_workers=1) # автоскан — по одному таймфрейму за раз
views = ThreadPoolExecutor(max_workers=2)         # листание графиков и примеров
busy = set()                                      # кто сейчас ждёт результат
busy_lock = threading.Lock()
chart_lock = threading.Lock()                     # matplotlib не умеет рисовать в двух потоках сразу


def start_job(chat_id, func, *args):
    """Запускает долгую задачу в фоне. Одному человеку — одна задача за раз."""
    with busy_lock:
        if chat_id in busy:
            send_message("⏳ Ещё работаю над прошлым запросом — пришлю, как закончу.", chat_id,
                         {"inline_keyboard": [[{"text": "✖️ Ок", "callback_data": "hide"}]]})
            return
        busy.add(chat_id)

    def run():
        try:
            func(chat_id, *args)
        except Exception as error:
            print("Ошибка в фоновой задаче:", error)
            traceback.print_exc()
            send_message("😕 Что-то пошло не так, попробуй ещё раз чуть позже.", chat_id,
                         {"inline_keyboard": [[{"text": "🏠 Меню", "callback_data": "home"}]]})
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


def market_example(df, item):
    """Находка с рынка → новый пример паттерна (аргументы для add_pattern) или None.
    Фигуры — эталон формы/линий, SMC, Range, флаги — отпечаток (fingerprint.py)."""
    example = example_from_item(df, item)
    if example is not None:
        return {"kind": example.get("kind"), "reference": example.get("reference"), "shape": example.get("shape")}
    if item.get("name") in FAMILY:
        return {"kind": item.get("kind"), "fingerprint": make_fingerprint(item["name"], df, item)}
    return None


# --- общие кнопки и экраны ------------------------------------------------------------------
HOME = {"text": "🏠 Меню", "callback_data": "home"}        # новое меню внизу чата (старое сообщение остаётся)
TO_MENU = {"text": "🏠 Меню", "callback_data": "menu"}     # меню на месте этого сообщения
BACK = {"text": "⬅️ Назад", "callback_data": "back"}
CLOSE = {"text": "✖️ Закрыть", "callback_data": "hide"}


def _grid(buttons, width=2):
    return [buttons[i:i + width] for i in range(0, len(buttons), width)]


def is_media(message):
    """Сообщение с картинкой или файлом: текст в нём заменить нельзя, только подпись."""
    return any(key in message for key in ("photo", "document"))


def show_view(chat_id, message_id, view, media=False):
    """Показывает экран на месте старого сообщения. Картинку в текст превратить нельзя —
    тогда старое сообщение удаляем и присылаем новое (в чате не копится мусор)."""
    text, keyboard = view
    if media and message_id is not None:
        delete_message(chat_id, message_id)
        message_id = None
    if message_id is None or not edit_message(chat_id, message_id, text, keyboard):
        return send_message(text, chat_id, keyboard)
    return message_id


def say(chat_id, message_id, text, keyboard=None):
    """Заменяет «⏳ Сканирую…» результатом; не вышло — новое сообщение."""
    if message_id is None or not edit_message(chat_id, message_id, text, keyboard):
        return send_message(text, chat_id, keyboard)
    return message_id


def outdated(chat_id):
    send_message("Эта находка устарела (бот перезапускался или было много сканов) — "
                 "запусти сканирование ещё раз.", chat_id, {"inline_keyboard": [[HOME]]})


# --- главное меню ----------------------------------------------------------------------------
def menu_keyboard(chat_id=None):
    """Главное меню — только действия, по два в ряд."""
    auto_on = chat_id is not None and get_autoscan(chat_id)
    return {"inline_keyboard": [
        [{"text": "🔍 Сканировать", "callback_data": "scan"},
         {"text": "🔎 Найти монету", "callback_data": "find"}],
        [{"text": "📂 Мои паттерны", "callback_data": "my"},
         {"text": "📊 История", "callback_data": "history"}],
        [{"text": "📚 Обучение", "callback_data": "learn"},
         {"text": "⚙️ Настройки", "callback_data": "settings"}],
        [{"text": "🔔 Автоскан: вкл" if auto_on else "🔕 Автоскан: выкл", "callback_data": "auto"},
         {"text": "❓ Помощь", "callback_data": "help"}],
    ]}


def menu_text(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    count = len({p["name"] for p in get_patterns(chat_id)})
    patterns = plural(count, "паттерн", "паттерна", "паттернов") if count else "паттернов пока нет"
    auto = "\n🔔 Автоскан включён — пришлю новые находки после закрытия свечи." if get_autoscan(chat_id) else ""
    return (f"🤖 Pattern Finder\n{patterns} · {tfs} · топ-{get_coins(chat_id)} монет{auto}\n"
            "Что делаем?")


def show_menu(chat_id, message_id=None, media=False):
    """Главное меню. message_id — открыть его на месте текущего сообщения."""
    return show_view(chat_id, message_id, (menu_text(chat_id), menu_keyboard(chat_id)), media)


# --- «📂 Мои паттерны»: все паттерны каталога → галерея примеров ------------------------------------
def my_patterns_view(chat_id):
    """Все паттерны каталога. ✅ — уже сохранённые (нажал — галерея картинок),
    без галочки — нажал и прислал скриншот, паттерн добавлен."""
    counts = {}
    for p in get_patterns(chat_id):
        counts[p["name"]] = counts.get(p["name"], 0) + 1

    def button(name):
        index = ALL_PATTERNS.index(name)
        if counts.get(name):
            return {"text": f"✅ {name} · {counts[name]}", "callback_data": f"gal:{index}:0"}
        return {"text": f"➕ {name}", "callback_data": f"pick:{index}"}

    rows = ([[{"text": "📐 Фигуры технического анализа", "callback_data": "noop"}]]
            + _grid([button(n) for n in TA_PATTERNS])
            + [[{"text": "💧 Smart Money (SMC)", "callback_data": "noop"}]]
            + _grid([button(n) for n in SMC_PATTERNS])
            + [[TO_MENU]])
    if counts:
        text = ("📂 Паттерны\n✅ — ищу (цифра — сколько твоих примеров): нажми, чтобы посмотреть, "
                "добавить или удалить.\n➕ — ещё не ищу: нажми и пришли свой пример — буду искать "
                "похожие на него.")
    else:
        text = ("📂 Выбери паттерн и пришли скриншот своего примера — я буду искать на рынке "
                "похожие именно на него. Нажми на любой ➕.")
    return text, {"inline_keyboard": rows}


def examples_of(chat_id, index):
    return [p for p in get_patterns(chat_id) if p["name"] == ALL_PATTERNS[index]]


def example_title(p):
    """Короткое описание примера: откуда он и что на нём."""
    if p.get("note", "").startswith("с рынка"):
        if p.get("fingerprint"):
            return f"{p['note']}\n{describe_fingerprint(p['fingerprint'])}"
        return p["note"]
    if p.get("fingerprint"):
        return f"скрин: {describe_fingerprint(p['fingerprint'])}"
    if p.get("builtin"):
        return "без скриншота — " + ("встроенный эталон" if p["name"] in SHAPE_PATTERNS else "по правилам")
    if p.get("kind"):
        return f"скрин: {KIND_NAMES.get(p['kind'], p['kind'])}"
    return "скрин"


def gallery_view(chat_id, index, pos):
    """Один пример паттерна: (картинка, подпись, кнопки) или None, если примеров не осталось."""
    name = ALL_PATTERNS[index]
    examples = examples_of(chat_id, index)
    if not examples:
        return None
    n = len(examples)
    pos = max(0, min(pos, n - 1))
    example = examples[pos]
    caption = f"📂 {name} — пример {pos + 1} из {n}\n{example_title(example)}\n\nКак ищу: {PATTERN_INFO[name]}"
    caption = caption[:CAPTION_LIMIT]
    rows = []
    if n > 1:
        rows.append([{"text": "◀️", "callback_data": f"gal:{index}:{(pos - 1) % n}"},
                     {"text": f"{pos + 1} / {n}", "callback_data": "noop"},
                     {"text": "▶️", "callback_data": f"gal:{index}:{(pos + 1) % n}"}])
    rows.append([{"text": "🗑 Удалить этот пример", "callback_data": f"gdel:{example['id']}"},
                 {"text": "📷 Добавить скрин", "callback_data": f"shot:{index}"}])
    rows.append([{"text": "👀 Пример с рынка", "callback_data": f"demo:{index}"},
                 {"text": "🗑 Удалить целиком", "callback_data": f"gall:{index}"}])
    rows.append([{"text": "⬅️ К паттернам", "callback_data": "my"}, TO_MENU])
    return example.get("photo"), caption, {"inline_keyboard": rows}


def placeholder_image(name=None):
    """Картинка для примера без файла: у включённого «без скриншота» — карточка паттерна
    (для фигур — встроенный эталон), у удалённого с диска файла — «картинки нет»."""
    index = ALL_PATTERNS.index(name) if name in ALL_PATTERNS else "none"
    path = os.path.join(PATTERNS_DIR, f"_card_{index}.png")
    if not os.path.exists(path):
        import matplotlib.pyplot as plt
        with chart_lock:
            fig, ax = plt.subplots(figsize=(5, 3))
            shape = template_shape(name) if name else None
            if shape:
                ax.plot(shape, color="royalblue", linewidth=3)
                ax.set_title(f"{name}: встроенный эталон", fontsize=13)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                text = f"{name}\nищу по правилам —\nскриншот не нужен" if name else "картинки нет\n(файл удалён)"
                ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
                ax.axis("off")
            fig.savefig(path)
            plt.close(fig)
    return path


def show_gallery(chat_id, index, pos, message_id=None, media=False):
    """Галерея: листаем примеры в одном сообщении. Из текстового экрана — удаляем его и
    присылаем картинку; в самой галерее — меняем картинку на месте."""
    view = gallery_view(chat_id, index, pos)
    if view is None:                                       # примеров не осталось — к списку
        show_view(chat_id, message_id, my_patterns_view(chat_id), media)
        return
    photo, caption, keyboard = view
    path = photo if photo and os.path.exists(photo) else placeholder_image(None if photo else ALL_PATTERNS[index])
    if media and message_id is not None and edit_photo(chat_id, message_id, path, caption, keyboard):
        return
    if message_id is not None:
        delete_message(chat_id, message_id)
    send_photo(path, caption, chat_id, keyboard)


def delete_example(chat_id, pattern_id, message_id):
    """🗑 в галерее: удалить пример и показать соседний на том же месте."""
    pattern = next((p for p in get_patterns(chat_id) if p["id"] == pattern_id), None)
    if pattern is None or pattern["name"] not in ALL_PATTERNS:
        show_view(chat_id, message_id, my_patterns_view(chat_id), media=True)
        return
    index = ALL_PATTERNS.index(pattern["name"])
    pos = [p["id"] for p in examples_of(chat_id, index)].index(pattern_id)
    removed = delete_pattern(chat_id, pattern_id)
    if removed:
        remove_file(removed["photo"])
    show_gallery(chat_id, index, pos, message_id, media=True)


def delete_all(chat_id, index):
    for p in examples_of(chat_id, index):
        removed = delete_pattern(chat_id, p["id"])
        if removed:
            remove_file(removed["photo"])


# --- паттерн, который ещё не ищем: объяснение, «включить без скрина», пример с рынка ----------------
def is_enabled(chat_id, name):
    return any(p["name"] == name for p in get_patterns(chat_id))


def intro_view(chat_id, index):
    """Экран паттерна: как бот его ищет и что можно сделать. Главное — ТВОЙ пример."""
    name = ALL_PATTERNS[index]
    count = len(examples_of(chat_id, index))
    text = f"🧩 {name}\n\nКак ищу: {PATTERN_INFO[name]}\n\n"
    if count:
        text += f"✅ Ищу. Твоих примеров: {count} — чем больше разных, тем точнее «похоже на твой пример»."
    else:
        text += ("📷 Пришли скриншот своего примера — буду искать на рынке ПОХОЖИЕ на него: "
                 "у каждого трейдера свой «правильный» паттерн.\n"
                 "Не знаешь, что сфотографировать? Открой «👀 Пример с рынка»: понравился — «💾 В мои примеры».")
    rows = [[{"text": "📷 Прислать свой пример", "callback_data": f"shot:{index}"}],
            [{"text": "👀 Пример с рынка", "callback_data": f"demo:{index}"}]]
    if not count:
        rows.append([{"text": "Искать без примера (по общему правилу)", "callback_data": f"on:{index}"}])
    rows.append([{"text": "⬅️ К паттернам", "callback_data": "my"}, TO_MENU])
    return text, {"inline_keyboard": rows}


def enable_pattern(chat_id, index):
    """Включить поиск без скриншота: по правилам или по встроенному эталону формы."""
    name = ALL_PATTERNS[index]
    if not any(p["name"] == name and p.get("builtin") for p in get_patterns(chat_id)):
        add_pattern(chat_id, name, None, note="без скриншота", shape=template_shape(name), builtin=True)
    return (f"✅ Ищу «{name}» по общему правилу. Пришли свой пример (📷) — тогда буду искать "
            "похожие именно на него и писать «похоже на твой пример N%».",
            saved_keyboard(index))


# «👀 Пример с рынка»: реальный случай паттерна на графике со свечами и объёмом
DEMO_COINS = 25                 # ищем на 25 самых торгуемых монетах
DEMO_INTERVALS = ("4h", "1d")
DEMO_SHOW = 12                  # листать можно 12 лучших случаев
DEMO_TTL = 1800                 # найденные случаи помним 30 минут (для всех пользователей)
demo_cache = {}                 # название паттерна → (когда искали, [(монета, ТФ, df, находка)])
demo_lock = threading.Lock()
demo_shown = {}                 # (chat_id, номер паттерна) → какой случай сейчас на экране


def demo_occurrences(name, fetch=None):
    """Случаи паттерна на топ-монетах: свежие и сильные первыми."""
    with demo_lock:
        cached = demo_cache.get(name)
        if cached and time.time() - cached[0] < DEMO_TTL:
            return cached[1]
    pattern = {"name": name, "shape": template_shape(name), "builtin": True}
    jobs, _ = build_history_jobs([pattern], {"fvg": "significant"})
    if not jobs:
        return []
    _, family_kind, run = jobs[0]
    found = []
    symbols = top_symbols(DEMO_COINS)
    for interval in DEMO_INTERVALS:
        for symbol, candles in fetch_many(symbols, interval, CANDLES, fetch or get_candles).items():
            if not candles or len(candles) < 150:
                continue
            df = candles_to_df(candles)
            for item in run(df):
                if family_kind == "block" and item.get("volume_high") is False:
                    continue                                   # пример OB — только с повышенным объёмом
                end = min(item["start"] + item.get("window", 1) - 1, len(df) - 1)
                fresh = end / len(df)                          # 1.0 — на последней свече
                found.append((item["score"] + 0.3 * fresh, symbol, interval, df,
                              {**item, "name": name, "builtin": True}, family_kind))
    found.sort(key=lambda f: f[0], reverse=True)
    result = [f[1:] for f in found[:DEMO_SHOW]]
    with demo_lock:
        demo_cache[name] = (time.time(), result)
    return result


def demo_view(name, occurrences, pos, index, enabled):
    """Подпись и кнопки для случая номер pos."""
    symbol, interval, df, item, family_kind = occurrences[pos]
    end = min(item["start"] + item.get("window", 1) - 1, len(df) - 1)
    end = min(item.get("end", end), len(df) - 1)
    when = f"{df['open_time'].iloc[end]:%d.%m.%Y %H:%M} UTC"
    detail = item.get("label") or ", ".join(
        t for t in (KIND_NAMES.get(item.get("kind")), quality_text(item)) if t)
    caption = (f"👀 {name} — пример с рынка ({pos + 1}/{len(occurrences)})\n"
               f"{symbol} {interval.upper()} · {when}\n{detail}\n\n"
               f"Что на картинке: {DEMO_LEGEND.get(family_kind, '')}")
    rows = [[{"text": "🔄 Другой пример", "callback_data": f"demo:{index}:{(pos + 1) % len(occurrences)}"},
             {"text": "📊 TradingView", "url": tradingview_link(symbol, interval)}]]
    if market_example(df, item) is not None:
        rows.append([{"text": "💾 В мои примеры", "callback_data": f"dsave:{index}"}])
    if not enabled:
        rows.append([{"text": "✅ Искать такое", "callback_data": f"on:{index}"}])
    rows.append([{"text": "⬅️ К паттернам", "callback_data": "my"}, TO_MENU])
    return caption[:CAPTION_LIMIT], {"inline_keyboard": rows}


def show_demo(chat_id, index, pos, message_id=None, media=False):
    name = ALL_PATTERNS[index]
    if not media and message_id is not None and (name not in demo_cache):
        edit_message(chat_id, message_id, f"⏳ Ищу пример «{name}» на {DEMO_COINS} монетах (4H, 1D)…")
    occurrences = demo_occurrences(name)
    if not occurrences:
        show_view(chat_id, message_id, (f"Сейчас не нашёл примеров «{name}» на {DEMO_COINS} монетах 😕 "
                                        "Попробуй позже.", intro_view(chat_id, index)[1]), media)
        return
    pos %= len(occurrences)
    demo_shown[(chat_id, index)] = occurrences[pos]
    symbol, interval, df, item, _ = occurrences[pos]
    path = f"demo_{chat_id}_{int(time.time() * 1000)}.png"
    with chart_lock:
        plot_example(df, symbol, interval, item, path, f"{symbol} {interval.upper()} — {name}")
    caption, keyboard = demo_view(name, occurrences, pos, index, is_enabled(chat_id, name))
    try:
        if media and message_id is not None and edit_photo(chat_id, message_id, path, caption, keyboard):
            return
        if message_id is not None:
            delete_message(chat_id, message_id)
        send_photo(path, caption, chat_id, keyboard)
    finally:
        remove_file(path)


def save_demo(chat_id, index):
    """«💾 В мои примеры»: случай с рынка становится примером паттерна (как свой скрин)."""
    shown = demo_shown.get((chat_id, index))
    if shown is None:
        send_message("Пример устарел — открой «👀 Пример с рынка» ещё раз.", chat_id, {"inline_keyboard": [[CLOSE]]})
        return
    symbol, interval, df, item, _ = shown
    example = market_example(df, item)
    path = f"{PATTERNS_DIR}/{chat_id}_{int(time.time() * 1000)}.png"
    with chart_lock:
        plot_example(df, symbol, interval, item, path, f"{symbol} {interval.upper()} — {item['name']}")
    add_pattern(chat_id, item["name"], path, note=f"с рынка: {symbol} {interval.upper()}", **example)
    count = len(examples_of(chat_id, index))
    send_message(f"💾 Сохранил как пример №{count} к «{item['name']}» — теперь ищу и похожие на него.",
                 chat_id, {"inline_keyboard": [[{"text": "🖼 Мои примеры", "callback_data": f"gal:{index}:{count - 1}"},
                                                CLOSE]]})


# --- настройки (всё листается в одном сообщении) ----------------------------------------------------
def settings_text(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    return (f"⚙️ Настройки поиска\n"
            f"Таймфреймы: {tfs}\n"
            f"Монеты: топ-{get_coins(chat_id)} по объёму\n\n"
            "🎯 Значимые FVG — импульсная свеча сломала структуру, дыра ≥ 0.75 ATR и объём выше "
            "обычного; зона отработана, когда свеча закрылась за её серединой. "
            "«Все» — любая дыра больше 0.5 ATR.\n"
            "🧩 Только комбинации — монеты, где одновременно нашлось 2+ твоих паттерна.\n"
            "🖼 Только похожие на мой пример — треугольники, клинья и каналы, похожие на твой "
            "рисунок хотя бы на 60% (наклон линий, сужение, путь цены).\n"
            "🎯 Учитывать мои оценки — отсеивать то, что ты отмечал 👎, и писать «тебе подходит N%».\n"
            "🔗 Только совпадения ТФ — монеты, где находки разных таймфреймов подтверждают "
            "друг друга. Нужно выбрать хотя бы 2 таймфрейма.\n"
            "📊 OB с объёмом — Order Block только если объём на свече блока или импульсе после неё "
            "≥ 1.5× среднего за 20 свечей.")


def settings_keyboard(chat_id):
    settings = get_settings(chat_id)
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    yes = lambda key: "да" if settings[key] else "нет"
    return {"inline_keyboard": [
        [{"text": f"⏱ ТФ: {tfs}", "callback_data": "tf"},
         {"text": f"🪙 Топ-{get_coins(chat_id)}", "callback_data": "coins"}],
        [{"text": "🎯 FVG: " + ("значимые" if settings["fvg"] == "significant" else "все"),
          "callback_data": "set:fvg"},
         {"text": f"🧩 Комбинации: {yes('combos_only')}", "callback_data": "set:combos_only"}],
        [{"text": f"🖼 Похожие на пример: {yes('similar_only')}", "callback_data": "set:similar_only"}],
        [{"text": f"🎯 Учитывать оценки 👍/👎: {yes('learn')}", "callback_data": "set:learn"}],
        [{"text": f"🔗 Только совпадения ТФ: {yes('confluence_only')}", "callback_data": "set:confluence_only"}],
        [{"text": f"📊 OB только с повышенным объёмом: {yes('ob_volume')}", "callback_data": "set:ob_volume"}],
        [TO_MENU],
    ]}


def settings_view(chat_id):
    return settings_text(chat_id), settings_keyboard(chat_id)


COINS_TEXT = ("🪙 Сколько самых торгуемых монет Binance проверять? Чем больше — тем дольше "
              "сканирование, но больше находок.")


def coins_keyboard(chat_id):
    chosen = get_coins(chat_id)
    row = [
        {"text": ("✅ " if n == chosen else "") + str(n), "callback_data": f"coins:{n}"}
        for n in COIN_OPTIONS
    ]
    return {"inline_keyboard": [row, [{"text": "⬅️ К настройкам", "callback_data": "settings"}]]}


TF_TEXT = "⏱ На каких таймфреймах искать? Нажимай, чтобы включить или выключить:"


def interval_keyboard(chat_id):
    chosen = get_intervals(chat_id)
    row = [
        {"text": ("✅ " if i in chosen else "▫️ ") + i.upper(), "callback_data": f"tf:{i}"}
        for i in ALL_INTERVALS
    ]
    return {"inline_keyboard": [row, [{"text": "✅ Готово", "callback_data": "tf_done"}]]}


# --- приём фото ------------------------------------------------------------------------
def saved_keyboard(index):
    return {"inline_keyboard": [
        [{"text": "📷 Ещё скрин", "callback_data": f"shot:{index}"},
         {"text": "📂 Мои паттерны", "callback_data": "my"}],
        [{"text": "🔍 Сканировать", "callback_data": "scan"}, TO_MENU],
    ]}


def retry_keyboard(index):
    return {"inline_keyboard": [[{"text": "📷 Прислать другой", "callback_data": f"shot:{index}"},
                                 {"text": "📂 Мои паттерны", "callback_data": "my"}], [TO_MENU]]}


def drawn_levels(path):
    """Горизонтальные линии, нарисованные на скрине (уровень блока, POC…) — в тех же
    «пикселях вверх», что и свечи со скрина: подсказка, какой именно паттерн ты имел в виду."""
    try:
        analysis = analyze_screenshot(path)
    except (OSError, ValueError):
        return []
    return [analysis.height - 1 - (ln.y1 + ln.y2) / 2 for ln in analysis.lines if abs(ln.angle) < 3]


def receive_example(chat_id, name, path):
    """SMC, Range, флаг / вымпел: читаю свечи со скрина, нахожу на них твой паттерн
    тем же правилом, что и на рынке, и снимаю отпечаток — с ним буду сравнивать находки."""
    index = ALL_PATTERNS.index(name)
    candles = extract_candles(path)
    fp, item = screenshot_fingerprint(name, candles, drawn_levels(path)) if candles is not None else (None, None)
    if fp is None:
        pattern_id = add_pattern(chat_id, name, path, note="скрин без отпечатка")
        seen = f"вижу {len(candles)} свечей, но" if candles is not None else "не смог прочитать свечи, и"
        send_message(f"😕 На скрине {seen} не нашёл «{name}» по правилу:\n{PATTERN_INFO[name]}\n\n"
                     "Скрин сохранил, но сравнивать с ним находки не смогу. Лучше пришли другой: "
                     "свечи крупно, видно 20–30 свечей до паттерна и несколько после, без индикаторов "
                     "(объём снизу — можно, я его прочитаю).", chat_id,
                     {"inline_keyboard": [[{"text": "📷 Прислать другой", "callback_data": f"shot:{index}"},
                                           {"text": "🗑 Удалить этот", "callback_data": f"bad:{pattern_id}"}],
                                          [{"text": "📂 Мои паттерны", "callback_data": "my"}, TO_MENU]]})
        return
    pattern_id = add_pattern(chat_id, name, path, note="скрин", kind=item.get("kind"), fingerprint=fp)
    debug_path = f"{PATTERNS_DIR}/{chat_id}_debug.png"
    with chart_lock:
        plot_example(candles, "", "", item, debug_path, f"Что я увидел на твоём скрине: {name}", dates=False)
    volume = "" if "volume" in candles else "\n(объёма на скрине не видно — сравниваю без него)"
    caption = (f"Вот что я увидел на твоём скрине ({len(candles)} свечей):\n"
               f"• {kind_text(item)}\n"
               f"• отпечаток: {describe_fingerprint(fp)}{volume}\n\n"
               "На рынке буду искать ПОХОЖИЕ на этот пример и писать «похоже на твой пример N%». "
               "Я правильно нашёл паттерн?")
    keyboard = {"inline_keyboard": [[
        {"text": "✅ Верно", "callback_data": f"ok:{pattern_id}"},
        {"text": "❌ Нет, удалить", "callback_data": f"bad:{pattern_id}"},
    ]]}
    send_photo(debug_path, caption[:CAPTION_LIMIT], chat_id, keyboard)
    remove_file(debug_path)


def receive_pattern_photo(chat_id, name, file_id):
    path = f"{PATTERNS_DIR}/{chat_id}_{int(time.time() * 1000)}.jpg"
    download_photo(file_id, path)
    index = ALL_PATTERNS.index(name)

    if name in FAMILY:                                    # SMC, Range, флаг, вымпел — по отпечатку
        receive_example(chat_id, name, path)
        return

    analysis = analyze_screenshot(path)
    if not analysis.ok:
        remove_file(path)
        send_message("Не смог разобрать график на картинке 😕\n\n" + TIPS, chat_id, retry_keyboard(index))
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
    caption += "\nВсё верно?"
    keyboard = {"inline_keyboard": [[
        {"text": "✅ Верно", "callback_data": f"ok:{pattern_id}"},
        {"text": "❌ Нет, удалить", "callback_data": f"bad:{pattern_id}"},
    ]]}
    send_photo(debug_path, caption, chat_id, keyboard)
    remove_file(debug_path)


TICKER = re.compile(r"^[A-Za-z0-9]{2,15}(/?USDT)?$", re.IGNORECASE)


WELCOME_KEYBOARD = {"inline_keyboard": [
    [{"text": "📂 Выбрать паттерны", "callback_data": "my"}, {"text": "🔍 Сканировать", "callback_data": "scan"}],
    [TO_MENU]]}


def send_welcome(chat_id):
    send_message(WELCOME, chat_id, WELCOME_KEYBOARD)


def handle_command(chat_id, text):
    """Команды из меню «/» Telegram. Возвращает True, если команда узнана."""
    command, _, argument = text.partition(" ")
    command = command.split("@")[0].lower()             # /scan@findercandles_bot — тоже /scan
    if command in ("/start", "/help"):
        send_welcome(chat_id)
    elif command == "/menu":
        show_menu(chat_id)
    elif command == "/patterns":
        show_view(chat_id, None, my_patterns_view(chat_id))
    elif command == "/scan":
        start_job(chat_id, run_scan)
    elif command == "/settings":
        show_view(chat_id, None, settings_view(chat_id))
    elif command == "/find":
        if argument.strip():
            find_coin(chat_id, argument.strip().replace(" ", ""))
        else:
            show_view(chat_id, None, (FIND_TEXT, {"inline_keyboard": [[TO_MENU]]}))
    else:
        return False
    return True


def handle_message(message):
    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()
    if first_visit(chat_id) and not text.startswith(("/start", "/help")):
        send_welcome(chat_id)                          # новичку — сначала объяснение

    if "photo" not in message:
        if text.startswith("/"):
            if not handle_command(chat_id, text):
                show_menu(chat_id)
            return
        ticker = text.replace(" ", "")
        if TICKER.match(ticker):
            find_coin(chat_id, ticker)           # написали тикер — ищем монету
            return
        show_menu(chat_id)
        return

    if chat_id not in waiting:
        send_message("Сначала выбери, какой это паттерн: «📂 Мои паттерны» → ➕ нужный паттерн.", chat_id,
                     {"inline_keyboard": [[{"text": "📂 Мои паттерны", "callback_data": "my"}, TO_MENU]]})
        return

    name = waiting.pop(chat_id)
    file_id = message["photo"][-1]["file_id"]
    receive_pattern_photo(chat_id, name, file_id)


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


# --- оценки 👍 / 👎 и обучение ---------------------------------------------------------------------
def coin_entries(coin):
    """Находки монеты с их графиком: [(таймфрейм, находка, df)]."""
    return [(f["interval"], it, f["df"]) for f in coin["frames"] if not f["history"] for it in f["items"]]


def feedback_key(symbol, tf, item, df):
    start = df["open_time"].iloc[item["start"]]
    return f"{symbol}|{tf}|{item['name']}|{int(start.timestamp())}"


def votes_of(chat_id):
    return {r["key"]: r["vote"] for r in get_feedback(chat_id)}


def record_vote(chat_id, coin, index, vote):
    """Запоминает оценку находки и рассказывает, если бот чему-то научился."""
    tf, item, df = coin_entries(coin)[index]
    name = item["name"]
    before = learn(get_feedback(chat_id)).get(name, {"records": [], "rules": []})
    record = {"key": feedback_key(coin["symbol"], tf, item, df), "at": time.time(),
              "symbol": coin["symbol"], "interval": tf, "name": name, "vote": vote,
              "features": features(df, item)}
    after = learn(add_feedback(chat_id, record)).get(name)
    count = len(after["records"])
    if after["rules"] and after["rules"] != before["rules"]:
        send_message(f"🎯 Научился по «{name}»: присылаю только если "
                     + "; ".join(rule_text(r) for r in after["rules"])
                     + ". Выключить — ⚙️ Настройки, посмотреть — 📚 Обучение.", chat_id)
    elif count == MIN_VOTES_FIT and len(before["records"]) < MIN_VOTES_FIT:
        send_message(f"🎯 По «{name}» уже {count} оценок — теперь пишу в алертах «тебе подходит N%».",
                     chat_id)


def single_vote(chat_id, coin):
    """Оценка монеты с ОДНОЙ находкой (1, -1 или None) — чтобы отметить 👍/👎 на кнопке."""
    entries = coin_entries(coin)
    if chat_id is None or len(entries) != 1:
        return None
    tf, item, df = entries[0]
    if "start" not in item:
        return None
    return votes_of(chat_id).get(feedback_key(coin["symbol"], tf, item, df))


def rating_keyboard(chat_id, card_id, coin):
    votes = votes_of(chat_id)
    rows = []
    for i, (tf, item, df) in enumerate(coin_entries(coin)):
        vote = votes.get(feedback_key(coin["symbol"], tf, item, df))
        quality = quality_text(item)
        name = item["name"] + (f" ({quality})" if quality else "")
        row = [{"text": ("✅ " if vote == 1 else "👍 ") + f"{tf.upper()} · {name}", "callback_data": f"rv:{card_id}:{i}:1"},
               {"text": "❌" if vote == -1 else "👎", "callback_data": f"rv:{card_id}:{i}:-1"}]
        if market_example(df, item) is not None:
            row.append({"text": "💾", "callback_data": f"ex:{card_id}:{i}"})
        rows.append(row)
    rows.append([{"text": "✅ Готово", "callback_data": "hide"}, HOME])
    return {"inline_keyboard": rows}


def vote_card(chat_id, card_id, vote, message_id=None):
    """👍 / 👎 под монетой: одна находка — запоминаем и отмечаем прямо на кнопке,
    несколько — спрашиваем по каждой отдельным сообщением."""
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    entries = coin_entries(coin)
    if len(entries) == 1:
        record_vote(chat_id, coin, 0, vote)
        if message_id is not None:                       # просмотрщик: ✅ на кнопке, без новых сообщений
            edit_keyboard(chat_id, message_id, card_keyboard(card_id, coin, chat_id))
            return
        text = "👍 Запомнил: такое тебе подходит." if vote > 0 else "👎 Запомнил: такое не присылать."
        keyboard = None
        if vote > 0 and market_example(entries[0][2], entries[0][1]) is not None:
            keyboard = {"inline_keyboard": [[{"text": "💾 Сохранить как пример", "callback_data": f"ex:{card_id}:0"}]]}
        send_message(text, chat_id, keyboard)
        return
    send_message(f"⭐ {coin['symbol']}: оцени каждую находку отдельно —\n"
                 "👍 подходит, 👎 не то, 💾 сохранить как пример паттерна.",
                 chat_id, rating_keyboard(chat_id, card_id, coin))


def rate_item(chat_id, card_id, index, vote, message_id):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    record_vote(chat_id, coin, index, vote)
    edit_keyboard(chat_id, message_id, rating_keyboard(chat_id, card_id, coin))


def save_example(chat_id, card_id, index):
    """«💾 Сохранить как пример»: находка с рынка становится ещё одним примером паттерна."""
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    tf, item, df = coin_entries(coin)[index]
    example = market_example(df, item)
    if example is None:
        send_message(f"«{item['name']}» сохранить как пример не получится — ставь 👍 / 👎, "
                     "так я узнаю, какие из них тебе подходят.", chat_id)
        return
    path = f"{PATTERNS_DIR}/{chat_id}_{int(time.time() * 1000)}.png"
    with chart_lock:                                  # картинка — чтобы было видно, что сохранено
        plot_example(df, coin["symbol"], tf, item, path, f"{coin['symbol']} {tf.upper()} — {item['name']}")
    add_pattern(chat_id, item["name"], path, note=f"с рынка: {coin['symbol']} {tf.upper()}", **example)
    count = sum(p["name"] == item["name"] for p in get_patterns(chat_id))
    record_vote(chat_id, coin, index, 1)                        # сохранил — значит, нравится
    index_name = ALL_PATTERNS.index(item["name"]) if item["name"] in ALL_PATTERNS else None
    keyboard = {"inline_keyboard": [[CLOSE]]}
    if index_name is not None:
        keyboard = {"inline_keyboard": [[{"text": "🖼 Посмотреть примеры", "callback_data": f"gal:{index_name}:{count - 1}"},
                                         CLOSE]]}
    send_message(f"💾 Сохранил как пример №{count} к «{item['name']}». Теперь ищу и похожие на него.",
                 chat_id, keyboard)


def learning_view(chat_id, note=""):
    keyboard = {"inline_keyboard": [[{"text": "♻️ Забыть все оценки", "callback_data": "fbreset"}],
                                    [TO_MENU]]}
    return note + learning_summary(get_feedback(chat_id)), keyboard


def user_judge(chat_id, settings):
    """Чему бот научился по оценкам этого человека (или None, если обучение выключено)."""
    return make_judge(get_feedback(chat_id)) if settings.get("learn", True) else None


# --- кнопка «🧩 Паттерны»: выбрать, что нанести на график ---------------------------------------
selections = {}          # (chat_id, card_id) → {"off": номера снятых галочек, "history": bool}


def coin_items(coin):
    """Все находки монеты по порядку: [(таймфрейм, находка)]."""
    return [(f["interval"], it) for f in coin["frames"] if not f["history"] for it in f["items"]]


def selection(chat_id, card_id):
    return selections.setdefault((chat_id, card_id), {"off": set(), "history": False})


def patterns_keyboard(chat_id, card_id, coin):
    sel = selection(chat_id, card_id)
    rows = []
    for i, (tf, item) in enumerate(coin_items(coin)):
        quality = quality_text(item)
        name = item["name"] + (f" ({quality})" if quality else "")
        mark = "▫️" if i in sel["off"] else "✅"
        rows.append([{"text": f"{mark} {tf.upper()} · {name}", "callback_data": f"sel:{card_id}:{i}"}])
    rows.append([{"text": "Выбрать все", "callback_data": f"sall:{card_id}"},
                 {"text": "Снять все", "callback_data": f"snone:{card_id}"}])
    rows.append([{"text": "🕘 Прошлые случаи этих паттернов: " + ("да" if sel["history"] else "нет"),
                  "callback_data": f"shist:{card_id}"}])
    rows.append([{"text": "📈 Разметка выбранного", "callback_data": f"psel:{card_id}"}])
    rows.append([CLOSE, HOME])
    return {"inline_keyboard": rows}


def show_patterns(chat_id, card_id, reply_to=None):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    send_message(f"🧩 {coin['symbol']}: что нанести на график TradingView?\n"
                 "Сними галочки с лишнего и нажми «Разметка выбранного».",
                 chat_id, patterns_keyboard(chat_id, card_id, coin), reply_to=reply_to)


PINE_HOWTO = ("❓ Как вставить разметку в TradingView:\n"
              "1. Нажми кнопку с таймфреймом под файлом — откроется нужный график Binance\n"
              "2. Удали старые индикаторы Pattern Finder (корзина рядом с названием)\n"
              "3. Внизу открой Pine Editor, удали всё и вставь код из файла\n"
              "4. Нажми «Add to chart»\n\n"
              "Красная линия — сопротивление, зелёная — поддержка, синие точки — касания; "
              "зелёные зоны — бычьи, красные — медвежьи; оранжевые точки EQH/EQL — ликвидность. "
              "Галочки (что показывать, прошлые случаи, ТФ) — шестерёнка индикатора → Inputs.")


def markup_keyboard(symbol, intervals):
    """Под файлом разметки — ссылки на график монеты в TradingView по каждому ТФ
    и выход в меню: после TradingView не надо листать чат вверх."""
    links = [{"text": f"📈 {symbol.removesuffix('USDT')} {tf.upper()}", "url": tradingview_link(symbol, tf)}
             for tf in intervals]
    return {"inline_keyboard": _grid(links, 3)
            + [[{"text": "❓ Как вставить", "callback_data": "pinehelp"}, HOME]]}


def send_markup(chat_id, coin, groups, code, filename, reply_to, **help_options):
    """Файл разметки ответом на карточку монеты: сверху цитата карточки (нажал — перешёл к ней),
    в подписи — что на графике, под файлом — ссылки на TradingView и меню."""
    intervals = list(dict.fromkeys(g[3] for g in groups if len(g) == 4)) or [coin["frames"][0]["interval"]]
    first = intervals[0]
    caption = f"📈 {coin['symbol']} — разметка для TradingView\n" + pine_help(
        coin["symbol"], first, groups, howto=False, **help_options)
    path = f"pine_{chat_id}_{int(time.time() * 1000)}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        send_document(path, caption[:CAPTION_LIMIT], chat_id, filename,
                      keyboard=markup_keyboard(coin["symbol"], intervals), reply_to=reply_to)
    finally:
        remove_file(path)


def change_selection(chat_id, card_id, message_id, action, index=None):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    sel = selection(chat_id, card_id)
    if action == "sel":
        sel["off"] ^= {index}                  # ^= — переключить: был в наборе — убрать, не было — добавить
    elif action == "sall":
        sel["off"] = set()
    elif action == "snone":
        sel["off"] = set(range(len(coin_items(coin))))
    elif action == "shist":
        sel["history"] = not sel["history"]
    edit_keyboard(chat_id, message_id, patterns_keyboard(chat_id, card_id, coin))


def send_selected_pine(chat_id, card_id, reply_to=None):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    sel = selection(chat_id, card_id)
    chosen = [entry for i, entry in enumerate(coin_items(coin)) if i not in sel["off"]]
    if not chosen:
        send_message("Ничего не выбрано — отметь хотя бы один паттерн ✅", chat_id)
        return
    send_action(chat_id)                         # «отправляет файл…» вверху чата
    groups = selected_markup(get_patterns(chat_id), coin, chosen, get_settings(chat_id))
    code = pine_script(coin["symbol"], chosen[0][0], groups, history_default=sel["history"])
    send_markup(chat_id, coin, groups, code, f"{coin['symbol']}_selected.pine.txt", reply_to,
                selected=True, history_default=sel["history"])


def send_pine(chat_id, card_id, reply_to=None):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    send_action(chat_id)                         # «отправляет файл…» вверху чата
    groups = full_markup(get_patterns(chat_id), coin, get_intervals(chat_id), get_settings(chat_id))
    code = pine_script(coin["symbol"], coin["frames"][0]["interval"], groups)
    send_markup(chat_id, coin, groups, code, f"{coin['symbol']}_patterns.pine.txt", reply_to)


# --- результаты: ОДИН список монет + ОДИН просмотрщик с листанием ◀️ ▶️ --------------------------------
CAPTION_LIMIT = 1000   # подпись к фото в Telegram — до 1024 символов
LIST_PAGE = 15         # монет на странице списка
ARROWS = {"up": " ↑", "down": " ↓"}
viewer_of = {}         # (chat_id, номер скана) → сообщение-просмотрщик (картинка монеты)
list_of = {}           # (chat_id, номер скана) → сообщение-список


def session_ids(coin, card_id):
    """Номера всех монет того же скана — для листания ◀️ ▶️ и «k / N»."""
    ids = coin.get("session") or [card_id]
    return ids if card_id in ids else [card_id]


def coin_icon(coin):
    if coin.get("history"):
        return "📊"
    if confluence_score(coin.get("confluence", [])) > 0:
        return "🔗"
    return "🔥" if any(len(f["items"]) > 1 for f in coin["frames"]) else "🔔"


def coin_line(coin, limit=90):
    """Одна строка списка: 🔗 SOL — 1H: FVG ↑ · 4H: Треугольник"""
    parts = []
    for f in coin["frames"]:
        names = ", ".join(it["name"] + ARROWS.get(direction(it), "") for it in f["items"])
        parts.append(f"{f['interval'].upper()}: {names}")
    text = f"{coin_icon(coin)} {coin['symbol'].removesuffix('USDT')} — " + " · ".join(parts)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def results_title(coins, head="🔍 Нашлось"):
    """Заголовок списка: сколько монет и что в них особенного."""
    combos = sum(any(len(f["items"]) > 1 for f in c["frames"]) for c in coins)
    multi = sum(len(c["frames"]) > 1 for c in coins)
    linked = sum(confluence_score(c.get("confluence", [])) > 0 for c in coins)
    title = f"{head} на {plural(len(coins), 'монете', 'монетах', 'монетах')}"
    extras = [f"🔗 совпадения ТФ: {linked}"] if linked else []
    if combos:
        extras.append(f"🔥 комбинации: {combos}")
    if multi:
        extras.append(f"на нескольких ТФ: {multi}")
    return title + (" (" + ", ".join(extras) + ")" if extras else "")


def results_view(chat_id, sid, page=0):
    """Список монет скана: строки с находками + кнопки «номер монета», по страницам."""
    first = find_card(chat_id, sid)
    if first is None:
        return ("Этот список устарел — запусти сканирование ещё раз.", {"inline_keyboard": [[TO_MENU]]})
    ids = session_ids(first, sid)
    pages = (len(ids) + LIST_PAGE - 1) // LIST_PAGE
    page = max(0, min(page, pages - 1))
    lines = [first.get("list_title", ""), "Нажми на монету — график откроется под списком, дальше листай ◀️ ▶️.", ""]
    buttons = []
    for n, card_id in enumerate(ids[page * LIST_PAGE:(page + 1) * LIST_PAGE], start=page * LIST_PAGE + 1):
        coin = find_card(chat_id, card_id)
        if coin is None:
            continue
        lines.append(f"{n}. {coin_line(coin)}")
        buttons.append({"text": f"{n}. {coin['symbol'].removesuffix('USDT')}", "callback_data": f"v:{card_id}"})
    rows = _grid(buttons, 3)
    if pages > 1:
        rows.append([{"text": "◀️", "callback_data": f"lp:{sid}:{(page - 1) % pages}"},
                     {"text": f"стр. {page + 1} / {pages}", "callback_data": "noop"},
                     {"text": "▶️", "callback_data": f"lp:{sid}:{(page + 1) % pages}"}])
    rows.append([HOME])
    return "\n".join(lines), {"inline_keyboard": rows}


def card_keyboard(card_id, coin, chat_id=None):
    """Кнопки просмотрщика монеты. chat_id — чтобы отметить уже поставленную 👍/👎."""
    rows = []
    ids = session_ids(coin, card_id)
    if len(ids) > 1:
        i, n = ids.index(card_id), len(ids)
        rows.append([{"text": "◀️", "callback_data": f"v:{ids[(i - 1) % n]}"},
                     {"text": f"{i + 1} / {n}", "callback_data": "noop"},
                     {"text": "▶️", "callback_data": f"v:{ids[(i + 1) % n]}"}])
    pine = {"text": "📈 Разметка", "callback_data": f"pine:{card_id}"}
    detail = {"text": "📄 Подробно", "callback_data": f"det:{card_id}"}
    frames = coin.get("frames") or []
    links = []
    if frames and frames[0].get("interval") in TV_INTERVALS:
        links = [{"text": "📊 TradingView", "url": tradingview_link(coin["symbol"], frames[0]["interval"])}]
    if coin.get("history"):
        rows.append([pine] + links)
        rows.append([detail])
    else:
        rows.append([{"text": "🧩 Паттерны", "callback_data": f"pats:{card_id}"}, pine] + links)
        vote = single_vote(chat_id, coin)
        rows.append([{"text": "✅ Подходит" if vote == 1 else "👍", "callback_data": f"vote:{card_id}:1"},
                     {"text": "❌ Не то" if vote == -1 else "👎", "callback_data": f"vote:{card_id}:-1"},
                     detail])
        if vote == 1:
            tf, item, df = coin_entries(coin)[0]
            if market_example(df, item) is not None:
                rows.append([{"text": "💾 Сохранить как пример", "callback_data": f"ex:{card_id}:0"}])
    bottom = [{"text": "📋 Список", "callback_data": f"list:{ids[0]}"}] if len(ids) > 1 else []
    rows.append(bottom + [HOME])
    return {"inline_keyboard": rows}


def short_item(item):
    """Находка коротко: «FVG ↑», «Канал (6 касаний)», «… 🎯80%». Цены зон — в «📄 Подробно»."""
    quality = quality_text(item)
    text = item["name"] + ARROWS.get(direction(item), "") + (f" ({quality})" if quality else "")
    if item.get("fit") is not None:
        text += f" 🎯{item['fit']:.0%}"
    return text


def viewer_caption(coin, card_id):
    """Короткая подпись просмотрщика: монета, место в списке, находки по ТФ, главное совпадение.
    Всё подробно — кнопка «📄 Подробно»."""
    ids = session_ids(coin, card_id)
    head = f"{coin_icon(coin)} {coin['symbol']}"
    if len(ids) > 1:
        head += f" · {ids.index(card_id) + 1}/{len(ids)}"
    if coin.get("found_at"):
        head += f" · скан {time.strftime('%H:%M', time.localtime(coin['found_at']))}"
    if coin.get("history"):
        body = coin.get("caption", "")
        if len(body) > CAPTION_LIMIT - len(head) - 40:
            body = body[:CAPTION_LIMIT - len(head) - 40].rsplit("\n", 1)[0] + "\n…(всё — «📄 Подробно»)"
        return f"{head}\n{body}"
    lines = [head]
    for f in coin["frames"]:
        lines.append(f"⏱ {f['interval'].upper()}: " + " · ".join(short_item(it) for it in f["items"]))
    found = coin.get("confluence", [])
    good = [c for c in found if c["weight"] > 0]
    bad = [c for c in found if c["weight"] < 0]
    lines += [f"🔗 {c['text']}" for c in good[:2]] + [f"⚠️ {c['text']}" for c in bad[:1]]
    if len(good) > 2 or len(bad) > 1:
        lines.append(f"…ещё совпадений: {len(good[2:]) + len(bad[1:])} — «📄 Подробно»")
    caption = "\n".join(lines)
    if len(caption) > CAPTION_LIMIT:
        caption = caption[:CAPTION_LIMIT - 30].rsplit("\n", 1)[0] + "\n…(всё — «📄 Подробно»)"
    return caption


def show_coin(chat_id, card_id, coin, target=None):
    """Рисует монету и показывает её в просмотрщике: target — сообщение, которое заменить
    (листание ◀️ ▶️), иначе — новое сообщение."""
    sid = session_ids(coin, card_id)[0]
    path = f"card_{chat_id}_{card_id}.png"
    with chart_lock:
        draw_coin(coin, path)
    caption, keyboard = viewer_caption(coin, card_id), card_keyboard(card_id, coin, chat_id)
    try:
        if target is not None and edit_photo(chat_id, target, path, caption, keyboard):
            viewer_of[(chat_id, sid)] = target
            return
        new_id = send_photo(path, caption, chat_id, keyboard)
        if new_id is not None:
            viewer_of[(chat_id, sid)] = new_id
    finally:
        remove_file(path)


def send_card(chat_id, card_id, coin):
    """Монета новым сообщением-просмотрщиком (поиск монеты, первая монета скана)."""
    show_coin(chat_id, card_id, coin)


def open_coin(chat_id, card_id, message_id=None, from_viewer=False):
    """Кнопка монеты: из просмотрщика (◀️ ▶️) — меняем картинку на месте; из списка —
    меняем просмотрщик под этим списком."""
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    sid = session_ids(coin, card_id)[0]
    target = message_id if from_viewer else viewer_of.get((chat_id, sid))
    show_coin(chat_id, card_id, coin, target)


def send_cards(chat_id, coins, title=None, edit_id=None):
    """Результаты скана компактно: ОДНО сообщение-список всех монет (edit_id — заменить им
    «⏳ Сканирую…») и ОДНО сообщение с графиком первой монеты, которое листается ◀️ ▶️.
    coins — результат group_by_coin, лучшие первыми."""
    if not coins:
        return
    now = time.time()
    for coin in coins:
        coin.setdefault("found_at", now)             # для поиска и подписи: «скан в 14:05»
    ids = remember_cards(chat_id, coins)
    for coin in coins:
        coin["session"] = ids
    sid = ids[0]
    coins[0]["list_title"] = title or results_title(coins)
    if len(coins) > 1:
        text, keyboard = results_view(chat_id, sid)
        list_of[(chat_id, sid)] = say(chat_id, edit_id, text, keyboard)
    else:
        say(chat_id, edit_id, coins[0]["list_title"])
    send_card(chat_id, sid, coins[0])


def show_list(chat_id, sid, viewer_id):
    """«📋 Список» в просмотрщике: список и просмотрщик переезжают вниз чата (старые удаляем)."""
    old = list_of.pop((chat_id, sid), None)
    if old is not None:
        delete_message(chat_id, old)
    delete_message(chat_id, viewer_id)
    viewer_of.pop((chat_id, sid), None)
    text, keyboard = results_view(chat_id, sid)
    list_of[(chat_id, sid)] = send_message(text, chat_id, keyboard)


def send_details(chat_id, card_id, reply_to=None):
    """«📄 Подробно»: полный текст находок, совпадений и ссылка на TradingView."""
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    parts = split_text(coin.get("caption", coin["symbol"]))
    for part in parts[:-1]:
        send_message(part, chat_id, reply_to=reply_to)
    send_message(parts[-1], chat_id, {"inline_keyboard": [[CLOSE, HOME]]}, reply_to=reply_to)


# --- поиск монеты ---------------------------------------------------------------------------------
FIND_BUTTONS = 12
FIND_TEXT = ("🔎 Напиши тикер монеты, например SOL, WLD или PEPEUSDT.\n"
             "Сначала ищу в последних результатах, а если её там нет — проверю её прямо сейчас.")


def stored_coins(chat_id):
    """Находки из последних сканов этого человека, свежие первыми: [(номер, монета)]."""
    with cards_lock:
        store = list(cards_by_chat.get(chat_id, {}).items())
    return [(card_id, coin) for card_id, coin in reversed(store) if not coin.get("history")]


def find_coin(chat_id, query):
    """Монета по тикеру: из последних результатов, по началу названия или проверка прямо сейчас."""
    base = query.upper().replace("/", "").removesuffix("USDT")
    symbol = f"{base}USDT"
    coins = stored_coins(chat_id)
    exact = [(cid, coin) for cid, coin in coins if coin["symbol"] == symbol]
    if exact:
        card_id, coin = exact[0]
        send_card(chat_id, card_id, coin)
        return
    similar = {}
    for card_id, coin in coins:                         # «SO» → SOL, SOLV…
        if coin["symbol"].startswith(base):
            similar.setdefault(coin["symbol"], card_id)
    if len(similar) > 1:
        buttons = [{"text": s.removesuffix("USDT"), "callback_data": f"v:{cid}"}
                   for s, cid in list(similar.items())[:FIND_BUTTONS]]
        send_message("🔎 Похожие монеты в последних результатах:", chat_id,
                     {"inline_keyboard": _grid(buttons, 4) + [[HOME]]})
        return
    if len(similar) == 1:
        symbol, card_id = next(iter(similar.items()))
        send_card(chat_id, card_id, find_card(chat_id, card_id))
        return
    start_job(chat_id, check_coin, symbol)


def check_coin(chat_id, symbol):
    """Монеты нет в результатах — проверяем её прямо сейчас на всех ТФ пользователя."""
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message("Сначала добавь хотя бы один паттерн 👆", chat_id, {"inline_keyboard": [[TO_MENU]]})
        return
    intervals = get_intervals(chat_id)
    if not get_candles(symbol, intervals[0], 5):
        send_message(f"Не нашёл {symbol} на Binance 😕 Проверь тикер (нужна пара к USDT).", chat_id,
                     {"inline_keyboard": [[TO_MENU]]})
        return
    tfs = ", ".join(i.upper() for i in intervals)
    wait_id = send_message(f"🔎 В последних результатах {symbol} нет — проверяю сейчас ({tfs})…", chat_id)
    # при поиске конкретной монеты показываем всё, что на ней есть: без «только комбинации»
    settings = {**get_settings(chat_id), "combos_only": False}
    cards, _ = scan_user(patterns, intervals, symbols=[symbol], settings=settings,
                         judge=user_judge(chat_id, settings))
    if not cards:
        say(chat_id, wait_id, f"На {symbol} сейчас нет твоих паттернов ({tfs}).", {"inline_keyboard": [[TO_MENU]]})
        return
    send_cards(chat_id, group_by_coin(cards), f"🔎 {symbol} — что нашлось сейчас ({tfs}):", wait_id)


def report_skipped(chat_id, skipped):
    if skipped:
        send_message("Пока не умею искать: " + ", ".join(skipped) + ". Детекторы скоро появятся.", chat_id)


NO_PATTERNS = ("Сначала добавь хотя бы один паттерн: «📂 Мои паттерны» → ➕.",
               {"inline_keyboard": [[{"text": "📂 Мои паттерны", "callback_data": "my"}, TO_MENU]]})


def run_scan(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message(NO_PATTERNS[0], chat_id, NO_PATTERNS[1])
        return
    intervals = get_intervals(chat_id)
    tfs = ", ".join(i.upper() for i in intervals)
    coins = get_coins(chat_id)
    wait_id = send_message(f"🔍 Сканирую топ-{coins} монет ({tfs}), {wait_text(coins, intervals)}…", chat_id)
    settings = get_settings(chat_id)
    cards, skipped = scan_user(patterns, intervals, symbols=top_symbols(coins), settings=settings,
                               judge=user_judge(chat_id, settings))
    report_skipped(chat_id, skipped)
    again = {"inline_keyboard": [[{"text": "🔍 Ещё раз", "callback_data": "scan"},
                                  {"text": "⚙️ Настройки", "callback_data": "settings"}], [TO_MENU]]}
    if not cards:
        text = "Совпадений сейчас нет."
        if settings["combos_only"]:
            text += " Включено «только комбинации» — одиночные находки не показываю (⚙️ Настройки)."
        say(chat_id, wait_id, text, again)
        return
    coins_found = group_by_coin(cards)
    if settings.get("confluence_only"):
        coins_found = [c for c in coins_found if confluence_score(c["confluence"]) > 0]
        if not coins_found:
            say(chat_id, wait_id, "Совпадений по таймфреймам сейчас нет (включено «🔗 Только совпадения ТФ»).",
                again)
            return
    send_cards(chat_id, coins_found, results_title(coins_found), wait_id)


def wait_text(coins, intervals):
    """Примерно сколько ждать: ~0.15 с на монету и таймфрейм."""
    seconds = coins * len(intervals) * 0.15
    return "это займёт до минуты" if seconds <= 60 else f"это займёт ~{round(seconds / 60)} мин"


def run_history(chat_id):
    patterns = get_patterns(chat_id)
    if not patterns:
        send_message(NO_PATTERNS[0], chat_id, NO_PATTERNS[1])
        return
    intervals = get_intervals(chat_id)
    coins = get_coins(chat_id)
    wait_id = send_message(f"📊 Считаю историю по топ-{coins} монет, {wait_text(coins * 2, intervals)}…", chat_id)
    reports, cards, skipped = history_user(patterns, intervals, symbols=top_symbols(coins),
                                           settings=get_settings(chat_id))
    report_skipped(chat_id, skipped)
    say(chat_id, wait_id, f"📊 История по топ-{coins} монет готова 👇")
    for report in reports:
        for part in split_text(report):
            send_message(part, chat_id)
    coins_found = group_by_coin(cards)
    if coins_found:
        send_cards(chat_id, coins_found, results_title(coins_found, "📊 Прошлые случаи"))
    else:
        send_message("Прошлых случаев не нашлось.", chat_id, {"inline_keyboard": [[HOME]]})


# --- кнопки ------------------------------------------------------------------------------
def background(func, *args):
    """Короткая фоновая работа без очереди «⏳ ещё работаю» (листание графиков)."""
    def run():
        try:
            func(*args)
        except Exception as error:
            print("Ошибка в фоне:", error)
            traceback.print_exc()
    views.submit(run)


def handle_callback(query):
    message = query["message"]
    chat_id = message["chat"]["id"]
    message_id = message["message_id"]
    media = is_media(message)
    data = query["data"]
    answer_callback(query["id"])

    if data == "noop":
        return

    if data == "home":                                  # меню внизу чата, сообщение с кнопкой остаётся
        show_menu(chat_id)
        return

    if data in ("back", "menu"):
        show_menu(chat_id, message_id, media)
        return

    if data == "hide":
        delete_message(chat_id, message_id)
        return

    if data in ("my", "add", "cancel"):
        waiting.pop(chat_id, None)
        show_view(chat_id, message_id, my_patterns_view(chat_id), media)
        return

    if data == "scan":
        start_job(chat_id, run_scan)
        return

    if data == "history":
        start_job(chat_id, run_history)
        return

    if data == "learn":
        show_view(chat_id, message_id, learning_view(chat_id), media)
        return

    if data == "fbreset":
        clear_feedback(chat_id)
        show_view(chat_id, message_id, learning_view(chat_id, "♻️ Забыл все оценки — учусь заново.\n\n"), media)
        return

    if data == "find":
        show_view(chat_id, message_id, (FIND_TEXT, {"inline_keyboard": [[TO_MENU]]}), media)
        return

    if data in ("settings", "tf_done"):
        show_view(chat_id, message_id, settings_view(chat_id), media)
        return

    if data == "tf":
        show_view(chat_id, message_id, (TF_TEXT, interval_keyboard(chat_id)), media)
        return

    if data == "coins":
        show_view(chat_id, message_id, (COINS_TEXT, coins_keyboard(chat_id)), media)
        return

    if data == "auto":                                  # меню обновляется на месте
        toggle_autoscan(chat_id)
        show_menu(chat_id, message_id, media)
        return

    if data == "help":
        show_view(chat_id, message_id, (WELCOME, WELCOME_KEYBOARD), media)
        return

    if data == "pinehelp":
        send_message(PINE_HOWTO, chat_id, {"inline_keyboard": [[CLOSE, HOME]]})
        return

    action, _, value = data.partition(":")

    if action == "tf":
        toggle_interval(chat_id, value, ALL_INTERVALS)
        edit_keyboard(chat_id, message_id, interval_keyboard(chat_id))
        return

    if action == "coins":
        set_coins(chat_id, int(value))
        show_view(chat_id, message_id, settings_view(chat_id), media)
        return

    if action == "set":
        settings = toggle_setting(chat_id, value)
        edit_keyboard(chat_id, message_id, settings_keyboard(chat_id))
        searchable = {p["name"] for p in get_patterns(chat_id) if p["name"] in SEARCHABLE}
        if value == "combos_only" and settings["combos_only"] and len(searchable) < 2:
            send_message("⚠️ Для комбинаций нужно минимум 2 разных паттерна — сейчас находок не будет. "
                         "Добавь ещё паттерн в «📂 Мои паттерны».", chat_id, {"inline_keyboard": [[CLOSE]]})
        return

    # --- результаты сканирования
    if action in ("v", "card"):
        background(open_coin, chat_id, int(value), message_id, media and action == "v")
        return

    if action == "lp":
        sid, _, page = value.partition(":")
        show_view(chat_id, message_id, results_view(chat_id, int(sid), int(page)))
        return

    if action == "list":
        show_list(chat_id, int(value), message_id)
        return

    if action == "det":
        send_details(chat_id, int(value), message_id)
        return

    if action == "pine":
        start_job(chat_id, send_pine, int(value), message_id)
        return

    if action == "pats":
        show_patterns(chat_id, int(value), message_id)
        return

    if action in ("sel", "sall", "snone", "shist"):
        card_id, _, index = value.partition(":")
        change_selection(chat_id, int(card_id), message_id, action, int(index) if index else None)
        return

    if action == "psel":
        start_job(chat_id, send_selected_pine, int(value), message_id)
        return

    if action == "vote":
        card_id, _, vote = value.partition(":")
        vote_card(chat_id, int(card_id), int(vote), message_id if media else None)
        return

    if action == "rv":
        card_id, index, vote = value.split(":")
        rate_item(chat_id, int(card_id), int(index), int(vote), message_id)
        return

    if action == "ex":
        card_id, _, index = value.partition(":")
        start_job(chat_id, save_example, int(card_id), int(index))
        return

    # --- паттерны и галерея примеров
    if action == "pick":                                # паттерн из списка: объяснение и действия
        show_view(chat_id, message_id, intro_view(chat_id, int(value)), media)
        return

    if action == "on":
        show_view(chat_id, message_id, enable_pattern(chat_id, int(value)), media)
        return

    if action == "demo":
        index, _, pos = value.partition(":")
        start_job(chat_id, show_demo, int(index), int(pos or 0), message_id, media)
        return

    if action == "dsave":
        start_job(chat_id, save_demo, int(value))
        return

    if action == "shot":
        name = ALL_PATTERNS[int(value)]
        waiting[chat_id] = name
        show_view(chat_id, message_id, (
            f"📷 Пришли скриншот паттерна «{name}».\n\n{TIPS}",
            {"inline_keyboard": [[{"text": "⬅️ К паттернам", "callback_data": "my"}, TO_MENU]]}), media)
        return

    if action in ("gal", "myp"):
        index, _, pos = value.partition(":")
        background(show_gallery, chat_id, int(index), int(pos or 0), message_id, media)
        return

    if action == "gdel":
        background(delete_example, chat_id, value, message_id)
        return

    if action == "gall":
        index = int(value)
        name = ALL_PATTERNS[index]
        count = len(examples_of(chat_id, index))
        edit_keyboard(chat_id, message_id, {"inline_keyboard": [
            [{"text": f"🗑 Да, удалить «{name}» ({count} шт.)", "callback_data": f"gyes:{index}"}],
            [{"text": "Отмена", "callback_data": f"gal:{index}:0"}]]})
        return

    if action == "gyes":
        delete_all(chat_id, int(value))
        show_view(chat_id, message_id, my_patterns_view(chat_id), media)
        return

    if action == "del":                        # старые кнопки: удалить один пример
        background(delete_example, chat_id, value, message_id)
        return

    # --- подтверждение нового скриншота
    if action == "ok":
        pattern = next((p for p in get_patterns(chat_id) if p["id"] == value), None)
        edit_keyboard(chat_id, message_id, {"inline_keyboard": [[{"text": "✅ Сохранено", "callback_data": "noop"}]]})
        if pattern is None or pattern["name"] not in ALL_PATTERNS:
            show_menu(chat_id)
            return
        send_message(f"✅ Сохранил «{pattern['name']}». Добавь ещё пример или запускай поиск.",
                     chat_id, saved_keyboard(ALL_PATTERNS.index(pattern["name"])))
        return

    if action == "bad":
        removed = delete_pattern(chat_id, value)
        edit_keyboard(chat_id, message_id, {"inline_keyboard": [[{"text": "🗑 Удалено", "callback_data": "noop"}]]})
        if removed is None:
            show_menu(chat_id)
            return
        remove_file(removed["photo"])
        index = ALL_PATTERNS.index(removed["name"]) if removed["name"] in ALL_PATTERNS else 0
        send_message(f"🗑 Удалил. Попробуй другой скриншот «{removed['name']}».\n\n{TIPS}",
                     chat_id, retry_keyboard(index))


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
    try:
        set_commands(COMMANDS)                           # меню «/» в Telegram
    except requests.RequestException as error:
        print("Не смог обновить команды бота:", error)
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
