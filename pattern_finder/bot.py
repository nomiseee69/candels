"""Telegram-бот: меню паттернов, приём скриншотов, проверка распознавания, сканирование.

Долгие задачи (сканирование, история, разметка, автоскан) идут в фоновых потоках,
поэтому бот отвечает на кнопки, пока ищет паттерны."""

import itertools
import re
import os
import secrets
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

from catalog import ALL_PATTERNS, LINE_PATTERNS, RULE_PATTERNS, SEARCHABLE, SHAPE_PATTERNS, SMC_PATTERNS, TA_PATTERNS
from notifier import (
    API, HTTP, PROXY, TOKEN, TV_INTERVALS, safe_error, answer_callback, answer_pre_checkout, api_post, delete_message, edit_keyboard, edit_message, edit_photo,
    send_action, send_document, send_message, send_photo, set_commands, tradingview_link,
)
from autoscan import Schedule, run_autoscan
from pine import pine_help, pine_script, short_name
from market import top_symbols
from scanner import (
    ALL_INTERVALS, CANDLES, COIN_OPTIONS, build_history_jobs, draw_coin, full_markup,
    group_by_coin, history_user, scan_user, selected_markup,
)
from texts import finding_name
from storage import (
    add_feedback, add_pattern, clear_feedback, delete_pattern, first_visit, get_autoscan, get_coins,
    get_feedback, get_intervals, get_pattern, get_patterns, get_settings, set_coins, set_spec, toggle_autoscan,
    update_pattern,
    toggle_interval, toggle_setting,
)
from chart import plot_example, plot_findings
from guide import COMMANDS, DEMO_LEGEND, GLOSSARY, HELP_TEXT, PATTERN_INFO, SCREENSHOT_GUIDE, WELCOME
from templates import template_shape
from feedback import (MIN_VOTES_FIT, features, item_window, learn, make_judge, rule_text,
                      what_changed, with_learning)
from feedback import summary as learning_summary
from similarity import SIMILAR_ENOUGH, example_from_item
from screenshot_query import screenshot_choices
from fingerprint import FAMILY, describe as describe_fingerprint, kind_text, make_fingerprint, range_from_box, screenshot_fingerprint
from vision import UnreliableScreenshot, extract_candles, extract_marks, extract_range
import billing
import storage
import brain
import language
import quality
import strength
import visual
from confluence import direction, score as confluence_score
from market import candles_to_df, fetch_many, get_candles
from texts import plural
from version import VERSION
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, kind_from_drawing
from vision import analyze_screenshot, describe_lines, draw_debug

PATTERNS_DIR = "patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

waiting = {}       # chat_id → название паттерна, для которого ждём фото
describing = {}    # chat_id → id примера, к которому ждём описание словами
pending_spec = {}  # chat_id → описание из подписи к скрину (language.parse), пока скрин разбирается
MAX_MESSAGE = 4000
KEEP_CARDS = 300   # сколько последних находок помним на пользователя (для кнопок под ними)

# --- фоновые задачи ------------------------------------------------------------------------
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_SCANS", "2"))   # сканов одновременно (больше — сервер не тянет)
jobs = ThreadPoolExecutor(max_workers=MAX_PARALLEL)  # сканирования и история пользователей
queued = 0                                        # сколько задач ждёт или идёт сейчас
autoscan_jobs = ThreadPoolExecutor(max_workers=1) # автоскан — по одному таймфрейму за раз
views = ThreadPoolExecutor(max_workers=2)         # листание графиков и примеров
busy = set()                                      # кто сейчас ждёт результат
busy_lock = threading.Lock()
chart_lock = threading.Lock()                     # matplotlib не умеет рисовать в двух потоках сразу


ALERT_EVERY = 600                                 # одну и ту же ошибку админу — не чаще раза в 10 минут
alerted = {}


def report_error(title, error):
    """Ошибка в консоль. Сетевые — одной строкой и без токена бота (он пароль),
    остальные — с подробностями, чтобы было видно, где сломалось, и сообщением админу в Telegram."""
    if isinstance(error, requests.RequestException):
        print(f"{title}: нет связи ({error.__class__.__name__}) — {safe_error(error)[:200]}")
        return
    print(f"{title}: {safe_error(error)}")
    traceback.print_exc()
    alert_admins(f"⚠️ {title}: {error.__class__.__name__}: {safe_error(error)[:300]}")


def alert_admins(text, now=None):
    now = now or time.time()
    key = text[:120]
    if now - alerted.get(key, 0) < ALERT_EVERY:
        return
    alerted[key] = now
    for admin in billing.admins():
        try:
            send_message(text, admin)
        except Exception:                                # уведомление не должно ронять бота
            pass


def start_job(chat_id, func, *args):
    """Запускает долгую задачу в фоне. Одному человеку — одна задача за раз."""
    with busy_lock:
        if chat_id in busy:
            send_message("Ещё работаю над прошлым запросом — пришлю, как закончу.", chat_id,
                         {"inline_keyboard": [[{"text": "✖️ Ок", "callback_data": "hide"}]]})
            return
        busy.add(chat_id)
        global queued
        queued += 1
        ahead = queued - MAX_PARALLEL
    if ahead > 0:
        send_message(f"Много запросов — ты в очереди, перед тобой {ahead}. Начну сразу, как освободится.",
                     chat_id)

    def run():
        try:
            func(chat_id, *args)
        except Exception as error:
            report_error("Ошибка в фоновой задаче", error)
            send_message("Что-то пошло не так, попробуй ещё раз чуть позже.", chat_id,
                         {"inline_keyboard": [[{"text": "🏠 Меню", "callback_data": "home"}]]})
        finally:
            global queued
            with busy_lock:
                busy.discard(chat_id)
                queued -= 1

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
    "Как сделать скриншот:\n"
    "• любая тема и масштаб, индикаторы лучше скрыть\n"
    "• отметь паттерн: зону — прямоугольником, уровень — линией, фигуру — двумя линиями\n"
    "• в подписи можно описать его словами: «OB, снял ликвидность, с объёмом»"
)


# --- Telegram API ----------------------------------------------------------------
def get_updates(offset):
    response = HTTP.get(
        f"{API}/getUpdates",
        params={"offset": offset, "timeout": 30},
        timeout=40,
    )
    return response.json()["result"]


def download_photo(file_id, path):
    info = HTTP.get(f"{API}/getFile", params={"file_id": file_id}, timeout=10).json()
    file_path = info["result"]["file_path"]
    content = HTTP.get(f"https://api.telegram.org/file/bot{TOKEN}/{file_path}", timeout=30).content
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
    send_message("Эта находка устарела: после перезапуска бот не помнит старые кнопки — "
                 "запусти сканирование ещё раз.", chat_id,
                 {"inline_keyboard": [[{"text": "🔍 Сканировать", "callback_data": "scan"}, HOME]]})


# --- главное меню ----------------------------------------------------------------------------
def menu_keyboard(chat_id=None):
    return {"inline_keyboard": [
        [{"text": "🔍 Поиск", "callback_data": "search"}, {"text": "📂 Мои паттерны", "callback_data": "my"}],
        [{"text": "🔔 Автоскан", "callback_data": "auto_section"}, {"text": "👤 Профиль", "callback_data": "profile"}],
    ]}


GUIDE_BUTTON = {"text": "📖 Как сделать скриншот", "callback_data": "tutorial:0"}
SEARCH_VIEW = ("🔍 Отправь фото графика с выделенной фигурой или зоной прямо в этот чат.",
               {"inline_keyboard": [[GUIDE_BUTTON],
                [{"text": "🔍 По сохранённым паттернам", "callback_data": "scan"}],
                [{"text": "🔎 Найти монету", "callback_data": "find"}, {"text": "📊 История", "callback_data": "history"}],
                [TO_MENU]]})


def autoscan_view(chat_id):
    on = get_autoscan(chat_id)
    action = "Выключить" if on else ("Включить" if billing.is_pro(chat_id) else "🔒 Включить с Pro")
    return ("🔔 Автоскан " + ("включён" if on else "выключен") +
            ". Проверяю сохранённые паттерны после закрытия свечи.",
            {"inline_keyboard": [[{"text": action, "callback_data": "auto"}],
                                 [{"text": "⚙️ Настройки", "callback_data": "settings"}], [TO_MENU]]})


def menu_text(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    count = len({p["name"] for p in get_patterns(chat_id)})
    patterns = plural(count, "паттерн", "паттерна", "паттернов") if count else "паттернов пока нет"
    auto = "\n🔔 Автоскан включён — пришлю новые находки после закрытия свечи." \
        if get_autoscan(chat_id) and billing.is_pro(chat_id) else ""
    return (f"📈 Pattern Finder · {plan_line(chat_id)}\n"
            f"{patterns} · {tfs} · топ-{get_coins(chat_id)} монет{auto}")


# --- подписка ---------------------------------------------------------------------------------------
DISCLAIMER = "Бот ищет паттерны и не даёт торговых сигналов; это не финансовый совет."


def plan_line(chat_id):
    """Тариф одной строкой — в меню."""
    st = billing.status(chat_id)
    if st["admin"]:
        return "Pro (админ)"
    if st.get("forever"):
        return "Pro навсегда"
    if st["pro"]:
        left = billing.days_left(st["until"])
        name = "Pro, пробный период" if st["trial"] else "Pro"
        tail = ", продлится сам" if st.get("recurring") else ""
        return f"{name} — осталось {left} дн. (до {billing.until_text(st['until'])}){tail}"
    return f"Free — сканов сегодня {billing.scans_left(chat_id)} из {billing.FREE['scans']}"


def profile_view(chat_id):
    """👤 Профиль: тариф, сколько осталось, что будет потом, лимиты, оплаты."""
    st = billing.status(chat_id)
    names = list(dict.fromkeys(p["name"] for p in get_patterns(chat_id)))
    lines = [f"👤 Профиль · ID {chat_id}", ""]
    if st["admin"]:
        lines.append("Тариф: 💎 Pro навсегда (ты админ)")
    elif st.get("forever"):
        lines.append("Тариф: ♾ Pro навсегда — подарок, без срока окончания")
    elif st["pro"]:
        left = billing.days_left(st["until"])
        lines.append("Тариф: " + ("🎁 Pro — пробный период" if st["trial"] else "💎 Pro"))
        lines.append(f"Действует до: {billing.until_text(st['until'])} — осталось {left} дн.")
        if st.get("recurring"):
            lines.append("Продление: автоматически звёздами каждые 30 дней (отключить — в настройках Telegram).")
        else:
            lines.append("После окончания: тариф Free, если не продлить.")
    else:
        lines.append("Тариф: Free")
        lines.append(f"Паттерны: ищу {min(len(names), billing.FREE['patterns'])} из {billing.FREE['patterns']} "
                     "возможных" + (f"; не ищу (нужен Pro): {', '.join(billing.locked_names(chat_id))}"
                                    if billing.locked_names(chat_id) else ""))
        lines.append(f"Сканов сегодня осталось: {billing.scans_left(chat_id)} из {billing.FREE['scans']}")
        lines.append("Автоскан и нейросеть — только в Pro.")
    lines.append(f"\nТвои паттерны: {', '.join(names) or 'пока нет'}")
    paid = billing.history(chat_id)[:3]
    if paid:
        lines.append("\nОплаты:")
        lines += [f"• {time.strftime('%d.%m.%Y', time.localtime(p['at']))} — "
                  f"{billing.PLANS.get(p['plan'], (0, 0, 0, p['plan']))[3]}, "
                  + (f"⭐ {p['amount']:.0f}" if p["currency"] == "XTR" else f"{p['amount']:.0f} ₽") for p in paid]
    button = "💎 Продлить" if st["pro"] and not st["admin"] else "💎 Купить Pro"
    rows = [[{"text": button if not st["admin"] else "💎 Подписка", "callback_data": "sub"}],
            [{"text": "⚙️ Настройки", "callback_data": "settings"}, {"text": "📚 Обучение", "callback_data": "learn"}],
            [{"text": "❓ Помощь", "callback_data": "help"}]]
    return "\n".join(lines), {"inline_keyboard": rows + [[TO_MENU]]}


def subscription_view(chat_id):
    st = billing.status(chat_id)
    month, year = billing.PLANS["month"], billing.PLANS["year"]
    lines = [f"💎 Подписка · {plan_line(chat_id)}\n",
             "Pro — все паттерны, без лимита сканов, автоскан и ответы на любые вопросы.",
             f"Free — {billing.FREE['patterns']} паттерна и {billing.FREE['scans']} скана в день.\n",
             f"⭐ Месяц — {month[1]} звёзд или {month[2]} ₽",
             f"⭐ Год — {year[1]} звёзд или {year[2]} ₽, на 20% дешевле\n",
             DISCLAIMER]
    rows = []
    try:
        links = billing.stars_links(api_post)
    except Exception as error:
        report_error("Ссылки на оплату звёздами", error)
        links = {}
    if links:
        rows.append([{"text": f"⭐ Месяц — {month[1]}", "url": links["month"]}] if "month" in links else [])
        rows.append([{"text": f"⭐ Год — {year[1]}", "url": links["year"]}] if "year" in links else [])
    if billing.yookassa_on():
        rows.append([{"text": f"💳 Месяц — {month[2]} ₽", "callback_data": "pay:month"},
                     {"text": f"💳 Год — {year[2]} ₽", "callback_data": "pay:year"}])
    rows = [r for r in rows if r]
    if not rows:
        lines.append("\nОплата сейчас недоступна — напиши администратору.")
    rows.append([TO_MENU])
    return "\n".join(lines), {"inline_keyboard": rows}


def paywall(chat_id, reason):
    texts = {
        "patterns": f"На Free можно искать до {billing.FREE['patterns']} разных паттернов.",
        "scans": f"На Free — {billing.FREE['scans']} скана в день, сегодня они закончились.",
        "autoscan": "Автоскан — функция Pro: бот сам присылает находки после закрытия каждой свечи.",
    }
    send_message(texts[reason] + "\n\nВ Pro ограничений нет.", chat_id,
                 {"inline_keyboard": [[{"text": "💎 Подписка", "callback_data": "sub"}, TO_MENU]]})


def pay_card(chat_id, plan):
    """💳 ЮKassa: ссылка на оплату + кнопка «проверить»."""
    url = billing.yookassa_create(chat_id, plan)
    if url is None:
        send_message("Не получилось создать оплату картой. Попробуй звёздами или позже.", chat_id,
                     {"inline_keyboard": [[{"text": "💎 Подписка", "callback_data": "sub"}]]})
        return
    send_message(f"💳 {billing.PLANS[plan][3]} — {billing.PLANS[plan][2]} ₽.\nОплати по кнопке; "
                 "я сам замечу оплату в течение минуты (или нажми «Я оплатил»).", chat_id,
                 {"inline_keyboard": [[{"text": "💳 Оплатить", "url": url}],
                                      [{"text": "🔄 Я оплатил — проверить", "callback_data": "paycheck"}]]})


def check_card_payments(chat_id=None):
    news = billing.yookassa_check(chat_id)
    for target, text in news:
        send_message(text, target, {"inline_keyboard": [[TO_MENU]]})
    if chat_id is not None and not news:
        send_message("Оплата пока не пришла. Если только что оплатил — подожди минуту, я пришлю сообщение.",
                     chat_id)


def handle_payment(message):
    """successful_payment: звёзды пришли (первый платёж или автопродление подписки)."""
    chat_id = message["chat"]["id"]
    text = billing.stars_paid(chat_id, message["successful_payment"])
    if text:
        send_message(text, chat_id, {"inline_keyboard": [[TO_MENU]]})
        alert_admins(f"💰 Оплата звёздами: {user_title(chat_id)}, {message['successful_payment'].get('total_amount')} ⭐", 0)
    else:
        alert_admins(f"⚠️ Платёж не засчитан (повтор или не та сумма): {chat_id} {message['successful_payment']}", 0)


def handle_refunded(message):
    """Звёзды вернулись пользователю (Telegram сообщил refunded_payment) — Pro снимаем."""
    chat_id = message["chat"]["id"]
    if billing.stars_refunded(chat_id, message["refunded_payment"]):
        send_message("↩️ Оплата возвращена — подписка Pro отменена.", chat_id, {"inline_keyboard": [[TO_MENU]]})
        alert_admins(f"↩️ Возврат звёзд: {user_title(chat_id)}", 0)


def handle_pre_checkout(query):
    ok = billing.check_pre_checkout(query)
    answer_pre_checkout(query["id"], ok, None if ok else "Этот счёт устарел — открой «💎 Подписка» ещё раз.")


# --- админ ------------------------------------------------------------------------------------------
ADMIN_HELP = ("🛠 Админ-панель — всё кнопками ниже. Команды тоже работают (вместо ID можно @username):\n"
              "/grant @user 30 — выдать Pro на 30 дней · /grant @user навсегда — без срока · /revoke @user — забрать Pro\n"
              "/user @user — карточка · /refund @user — вернуть последнюю оплату звёздами\n"
              "/broadcast текст — сообщение всем · /backup — копия базы")
USERS_PAGE = 8
broadcast_draft = {}            # админ → текст рассылки, ждёт подтверждения
writing_broadcast = set()       # админ нажал «📣 Рассылка» — следующее сообщение = текст


def stars_balance():
    """Сколько звёзд на балансе бота (Bot API getMyStarBalance) или None."""
    answer = api_post("getMyStarBalance", {})
    if answer.get("ok"):
        return answer["result"].get("amount")
    return None


def find_user_id(word):
    """«123456» или «@username» → chat_id или None."""
    word = (word or "").strip()
    if word.lstrip("-").isdigit():
        return int(word)
    name = word.lstrip("@").lower()
    if not name:
        return None
    for key, user in storage.load_users().items():
        if (user.get("username") or "").lower() == name:
            return int(key)
    return None


def remember_username(message):
    """Запоминаем @username — чтобы админ мог писать /grant @user, а в списке было видно, кто есть кто."""
    sender = message.get("from") or {}
    chat_id = message.get("chat", {}).get("id")
    if not chat_id or sender.get("id") != chat_id:
        return
    name = sender.get("username") or sender.get("first_name") or ""
    if name and storage.get_user(chat_id).get("username") != name:
        storage.update_user(chat_id, lambda u: u.__setitem__("username", name))


def user_title(key, user=None):
    user = user if user is not None else storage.get_user(key)
    name = user.get("username")
    return f"@{name}" if name else str(key)


def admin_view(chat_id):
    balance = stars_balance()
    wallet = (f"\n⭐ На балансе бота: {balance} звёзд — это твои деньги; вывод: профиль бота в Telegram → "
              "«Баланс» / Fragment" if balance is not None else "")
    rows = [[{"text": "👥 Пользователи", "callback_data": "adm:users:0"},
             {"text": "📣 Рассылка", "callback_data": "adm:bc"}],
            [{"text": "💾 Копия базы", "callback_data": "adm:backup"},
             {"text": "🔄 Обновить", "callback_data": "adm:home"}], [TO_MENU]]
    return billing.stats() + wallet + "\n\n" + ADMIN_HELP, {"inline_keyboard": rows}


def users_view(page=0):
    users = sorted(storage.load_users().items(), key=lambda kv: -((kv[1].get("sub") or {}).get("until") or 0))
    part = users[page * USERS_PAGE:(page + 1) * USERS_PAGE]
    rows = []
    for key, user in part:
        st = billing.status(int(key))
        mark = "👑" if st["admin"] else "♾" if st.get("forever") else (
            "💎" if st["pro"] and not st["trial"] else "🎁" if st["pro"] else "🆓")
        until = f" до {billing.until_text(st['until'])}" if st["pro"] and st["until"] else ""
        rows.append([{"text": f"{mark} {user_title(key, user)}{until}", "callback_data": f"adm:u:{key}"}])
    nav = []
    if page > 0:
        nav.append({"text": "◀️", "callback_data": f"adm:users:{page - 1}"})
    if (page + 1) * USERS_PAGE < len(users):
        nav.append({"text": "▶️", "callback_data": f"adm:users:{page + 1}"})
    rows += [nav] if nav else []
    rows.append([{"text": "⬅️ Админ-панель", "callback_data": "adm:home"}])
    text = (f"👥 Пользователи ({len(users)}). 👑 админ · 💎 платный Pro · 🎁 пробный · 🆓 Free.\n"
            "Нажми на человека — выдать или забрать Pro, вернуть оплату.")
    return text, {"inline_keyboard": rows}


def user_card(key):
    key = int(key)
    user = storage.get_user(key)
    st = billing.status(key)
    names = sorted({p["name"] for p in user.get("patterns", [])})
    plan = ("👑 админ" if st["admin"] else "♾ Pro навсегда" if st.get("forever") else
            (("🎁 пробный Pro" if st["trial"] else "💎 Pro") + f" до {billing.until_text(st['until'])}, "
             f"осталось {billing.days_left(st['until'])} дн." if st["pro"] else "🆓 Free"))
    paid = billing.history(key)[:5]
    lines = [f"👤 {user_title(key, user)} · ID {key}", f"Тариф: {plan}",
             f"Паттерны: {', '.join(names) or '—'}",
             f"Автоскан: {'вкл' if user.get('autoscan') else 'выкл'} · оценок 👍/👎: {len(user.get('feedback', []))}"]
    if paid:
        lines.append("Оплаты:")
        lines += [f"• {time.strftime('%d.%m.%Y', time.localtime(p['at']))} {p['plan']} — "
                  + (f"⭐ {p['amount']:.0f}" if p["currency"] == "XTR" else f"{p['amount']:.0f} ₽")
                  + (" (возвращена)" if p.get("status") == "refunded" else "") for p in paid]
    rows = [[{"text": "🎁 +30 дней Pro", "callback_data": f"adm:grant:{key}:30"},
             {"text": "🎁 +365 дней", "callback_data": f"adm:grant:{key}:365"}],
            [{"text": "♾ Pro навсегда", "callback_data": f"adm:grant:{key}:forever"}],
            [{"text": "⛔ Забрать Pro (сразу Free)", "callback_data": f"adm:revoke:{key}"}]]
    if any(p["currency"] == "XTR" and p.get("status") != "refunded" for p in paid):
        rows.append([{"text": "↩️ Вернуть последнюю оплату звёздами", "callback_data": f"adm:refund:{key}"}])
    rows.append([{"text": "⬅️ Список", "callback_data": "adm:users:0"},
                 {"text": "🛠 Панель", "callback_data": "adm:home"}])
    return "\n".join(lines), {"inline_keyboard": rows}


def admin_grant(admin_id, key, days):
    if days is None:                                     # навсегда
        billing.grant_forever(key)
        send_message("🎁 Тебе выдан Pro навсегда — все функции без ограничений. Спасибо, что с нами!", key)
        return f"✅ {user_title(key)}: ♾ Pro навсегда"
    until = billing.extend(key, days, source="admin")
    send_message(f"🎁 Тебе выдан Pro до {billing.until_text(until)}.", key)
    return f"✅ {user_title(key)}: Pro до {billing.until_text(until)}"


def admin_refund(key):
    """Вернуть последнюю (не возвращённую) оплату звёздами и снять Pro."""
    stars = [p for p in billing.history(key) if p["currency"] == "XTR" and p.get("status") != "refunded"]
    if not stars:
        return "У него нет оплат звёздами, которые можно вернуть."
    answer = api_post("refundStarPayment", {"user_id": int(key), "telegram_payment_charge_id": stars[0]["id"]})
    if not answer.get("ok"):
        return f"Не получилось вернуть: {answer.get('description', 'ошибка Telegram')}"
    storage.set_payment_status(stars[0]["id"], "refunded")
    billing.revoke(int(key))
    return f"↩️ Вернул {stars[0]['amount']:.0f} ⭐ пользователю {user_title(key)}, Pro снят."


def admin_callback(chat_id, message_id, value, media):
    """Кнопки админ-панели (adm:…). Каждый раз проверяем, что нажал админ."""
    if not billing.is_admin(chat_id):
        return
    parts = value.split(":")
    action = parts[0]
    if action == "home":
        show_view(chat_id, message_id, admin_view(chat_id), media)
    elif action == "users":
        show_view(chat_id, message_id, users_view(int(parts[1] or 0)), media)
    elif action == "u":
        show_view(chat_id, message_id, user_card(parts[1]), media)
    elif action == "grant":
        text = admin_grant(chat_id, int(parts[1]), None if parts[2] == "forever" else int(parts[2]))
        show_view(chat_id, message_id, user_card(parts[1]), media)
        send_message(text, chat_id)
    elif action == "revoke":
        billing.revoke(int(parts[1]))
        show_view(chat_id, message_id, user_card(parts[1]), media)
        send_message(f"⛔ {user_title(parts[1])}: Pro снят, теперь Free.", chat_id)
    elif action == "refund":
        send_message(f"Вернуть последнюю оплату звёздами пользователю {user_title(parts[1])} и снять Pro?", chat_id,
                     {"inline_keyboard": [[{"text": "↩️ Да, вернуть", "callback_data": f"adm:refundok:{parts[1]}"},
                                           {"text": "Отмена", "callback_data": "hide"}]]})
    elif action == "refundok":
        send_message(admin_refund(parts[1]), chat_id)
    elif action == "backup":
        path = storage.backup()
        send_message(f"Копия базы: {path}" if path else "Базы ещё нет.", chat_id)
    elif action == "bc":
        writing_broadcast.add(chat_id)
        send_message("📣 Напиши следующим сообщением текст рассылки — я покажу, как он выглядит, и спрошу, "
                     "отправлять ли всем.", chat_id, {"inline_keyboard": [[{"text": "Отмена", "callback_data": "adm:bcno"}]]})
    elif action == "bcok":
        text = broadcast_draft.pop(chat_id, None)
        if text:
            background(broadcast, chat_id, text)
    elif action == "bcno":
        writing_broadcast.discard(chat_id)
        broadcast_draft.pop(chat_id, None)
        send_message("Рассылка отменена.", chat_id)


def draft_broadcast(chat_id, text):
    broadcast_draft[chat_id] = text
    count = len(storage.load_users())
    send_message(f"Так увидят {count} пользователей:\n\n{text}", chat_id,
                 {"inline_keyboard": [[{"text": f"✅ Отправить всем ({count})", "callback_data": "adm:bcok"},
                                       {"text": "Отмена", "callback_data": "adm:bcno"}]]})


def admin_command(chat_id, command, argument):
    """Команды админа. True — команда админская (обработана). Не админ — команды «не существует»."""
    if command not in ("/admin", "/grant", "/revoke", "/user", "/broadcast", "/backup", "/refund", "/users"):
        return False
    if not billing.is_admin(chat_id):
        return False
    args = [a.strip(",;") for a in argument.replace(",", " ").split() if a.strip(",;")]
    who = find_user_id(args[0]) if args else None
    if command == "/admin":
        show_view(chat_id, None, admin_view(chat_id))
    elif command == "/users" or (command in ("/user", "/grant", "/revoke", "/refund") and who is None):
        if args:
            send_message(f"Не нашёл пользователя {args[0]} — он должен хоть раз написать боту. Выбери из списка:",
                         chat_id)
        show_view(chat_id, None, users_view(0))         # без ID — список пользователей с кнопками
    elif command == "/user":
        show_view(chat_id, None, user_card(who))
    elif command == "/grant":
        # «/grant @a @b @c навсегда», «/grant @a 90», «/grant @a» (30 дней)
        words = [a.lower().strip(",.") for a in args]
        forever = any(w in ("навсегда", "forever", "∞", "бессрочно", "вечно", "0") for w in words)
        number = next((int(w) for w in words[1:] if w.isdigit() and w != "0" and len(w) <= 4), None)
        days = None if forever else number or 30
        person = lambda a: a.startswith("@") or (a.isdigit() and len(a) > 4)     # @ник или Telegram ID
        others = [a for a in args[1:] if person(a)]
        people = [who] + [find_user_id(a) for a in others]
        missing = [a for a in others if find_user_id(a) is None]
        for key in dict.fromkeys(p for p in people if p is not None):
            send_message(admin_grant(chat_id, key, days), chat_id)
        if missing:
            send_message("Не нашёл: " + ", ".join(missing) + " — пусть сначала напишут боту /start.", chat_id)
    elif command == "/revoke":
        billing.revoke(who)
        send_message(f"⛔ {user_title(who)}: Pro снят, теперь Free.", chat_id)
    elif command == "/refund":
        send_message(admin_refund(who), chat_id)
    elif command == "/broadcast":
        if argument.strip():
            draft_broadcast(chat_id, argument.strip())
        else:
            admin_callback(chat_id, None, "bc", False)
    elif command == "/backup":
        path = storage.backup()
        send_message(f"Копия базы: {path}" if path else "Базы ещё нет.", chat_id)
    return True


def broadcast(admin_id, text):
    sent = 0
    for key in storage.load_users():
        try:
            if send_message(text, int(key)) is not None:
                sent += 1
        except Exception:
            pass
        time.sleep(0.05)                                 # Telegram: не больше ~30 сообщений в секунду
    send_message(f"📣 Разослано: {sent}", admin_id)


def show_menu(chat_id, message_id=None, media=False):
    """Главное меню. message_id — открыть его на месте текущего сообщения."""
    leave_photo_flow(chat_id)
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
        text = "📂 Паттерны\n✅ — ищу (цифра — твои примеры) · ➕ — добавить"
    else:
        text = "📂 Выбери паттерн (➕) и пришли скриншот своего примера — буду искать похожие."
    return text, {"inline_keyboard": rows}


def examples_of(chat_id, index):
    return [p for p in get_patterns(chat_id) if p["name"] == ALL_PATTERNS[index]]


def example_title(p):
    """Короткое описание примера: откуда он и что на нём."""
    if p.get("note", "").startswith("с рынка"):
        return p["note"]
    if p.get("fingerprint"):
        return "твой скрин"
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
    described = understood_text(example.get("spec"))
    caption = (f"📂 {name} — пример {pos + 1} из {n}\n{example_title(example)}\n\n"
               + (f"{described}\n" if described else "") + f"Как ищу: {PATTERN_INFO[name]}")
    caption = caption[:CAPTION_LIMIT]
    rows = []
    if n > 1:
        rows.append([{"text": "◀️", "callback_data": f"gal:{index}:{(pos - 1) % n}"},
                     {"text": f"{pos + 1} / {n}", "callback_data": "noop"},
                     {"text": "▶️", "callback_data": f"gal:{index}:{(pos + 1) % n}"}])
    rows.append([{"text": "✏️ Описать словами", "callback_data": f"describe:{example['id']}"}])
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
        text += f"✅ Ищу. Твоих примеров: {count}."
    else:
        text += "📷 Пришли скриншот своего примера — буду искать похожие. Нет примера? «👀 Пример с рынка»."
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
            "похожие именно на него.",
            saved_keyboard(index))


# «👀 Пример с рынка»: реальный случай паттерна на графике со свечами и объёмом
DEMO_COINS = 25                 # ищем на 25 самых торгуемых монетах
DEMO_INTERVALS = ("4h", "1d")
DEMO_SHOW = 12                  # листать можно 12 лучших случаев
DEMO_TTL = 1800                 # найденные случаи помним 30 минут (для всех пользователей)
demo_cache = {}                 # название паттерна → (когда искали, [(монета, ТФ, df, находка)])
demo_lock = threading.Lock()
demo_shown = {}                 # (chat_id, номер паттерна) → какой случай сейчас на экране


TEXTBOOK = {   # пример с рынка — как из учебника: эти критерии эксперта должны быть ярко выражены
    "zone": {"impulse": 0.6, "volume": 0.3, "imbalance": 0.5},
    "structure": {"close_beyond": 0.4, "body": 0.5, "leg": 0.4},
    "eq": {"touches": 0.3, "tight": 0.5, "untapped": 0.5},
}


def textbook(item):
    """Эталонный ли это случай для «👀 Пример с рынка»: сильный по оценке эксперта и с главными
    признаками паттерна (у OB — резкий импульс сразу после блока, объём и FVG)."""
    if item.get("grade") == "weak" or item.get("textbook") is False:
        return False                                   # блок без ликвидности / поглощения / объёма
    need = TEXTBOOK.get(item.get("group"), {})
    facts = item.get("factors") or {}
    if str(item.get("kind", "")).endswith(("_breaker", "_mitigation")):
        need = {k: v for k, v in need.items() if k != "imbalance"}
    return all(facts.get(k, 1) >= v for k, v in need.items())


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
                power = strength.evaluate(df, item) or 0.0
                end = min(item["start"] + item.get("window", 1) - 1, len(df) - 1)
                fresh = end / len(df)                          # 1.0 — на последней свече
                # сначала эталонные случаи (как из учебника), среди них — самые сильные
                found.append((float(textbook(item)) + power + 0.1 * fresh, symbol, interval, df,
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
    detail = item.get("label") or KIND_NAMES.get(item.get("kind")) or finding_name(item)
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
    send_message(f"✅ Сохранил как пример №{count} к «{item['name']}» — теперь ищу и похожие на него.",
                 chat_id, {"inline_keyboard": [[{"text": "🖼 Мои примеры", "callback_data": f"gal:{index}:{count - 1}"},
                                                CLOSE]]})


# --- настройки (всё листается в одном сообщении) ----------------------------------------------------
def settings_text(chat_id):
    tfs = ", ".join(i.upper() for i in get_intervals(chat_id))
    return f"⚙️ Настройки\nТаймфреймы: {tfs} · топ-{get_coins(chat_id)} монет\n\nНажми, чтобы изменить."


SETTINGS_HELP = (
    "❓ Что значат настройки:\n"
    "🎯 FVG — только сильные или все\n"
    "🧩 Комбинации — только монеты, где нашлось 2+ паттерна\n"
    "🖼 Похожие на пример — только похожие на твои скрины\n"
    "🎯 Оценки 👍/👎 — подстраиваюсь под твой вкус\n"
    "🔗 Совпадения ТФ — только когда разные таймфреймы подтверждают друг друга\n"
    "📊 Эталонные OB — только блоки со снятием ликвидности и объёмом\n"
    "🧠 Строгость — сколько слабых находок отсеивать"
)


QUALITY_NAMES = {"normal": "обычная", "strict": "строгая", "off": "выкл"}


def quality_status():
    on = [quality.FAMILY_TEXT[f] for f, m in quality.MODELS.items() if m.get("on")]
    if on:
        return "обучена для " + ", ".join(on)
    return "не обучена (эксперт работает и без неё; python setup_ai.py её дообучит)"


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
        [{"text": f"📊 OB / Breaker только эталонные: {yes('ob_volume')}", "callback_data": "set:ob_volume"}],
        [{"text": f"🧠 Строгость отбора: {QUALITY_NAMES.get(settings.get('quality', 'normal'), 'обычная')}",
          "callback_data": "set:quality"}],
        [{"text": "❓ Что это значит", "callback_data": "sethelp"}, TO_MENU],
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


DESCRIBE_TEXT = (
    "✏️ Опиши словами, каким должен быть твой паттерн — как другу-трейдеру.\n\n"
    "Например: «бычий OB на 4H, обязательно снял ликвидность, с объёмом, желательно в дисконте, "
    "без FVG»."
)
BRAIN_TIP = "\n\n🧠 Пиши как удобно, своими словами и сленгом — я пойму."


def example_keyboard(pattern_id, item=None):
    rows = [[{"text": "✅ Верно", "callback_data": f"ok:{pattern_id}"},
             {"text": "❌ Нет, удалить", "callback_data": f"bad:{pattern_id}"}]]
    kind = str((item or {}).get("kind", ""))
    if kind.startswith(("bullish", "bearish")):        # бот понял направление наоборот — поправь одной кнопкой
        other = "медвежий" if kind.startswith("bullish") else "бычий"
        rows.append([{"text": f"🔄 Нет, это {other}", "callback_data": f"flipdir:{pattern_id}"}])
    rows.append([{"text": "✏️ Описать словами", "callback_data": f"describe:{pattern_id}"}])
    return {"inline_keyboard": rows}


def flip_direction(chat_id, message_id, pattern_id):
    """«🔄 Нет, это медвежий»: меняем направление примера (и его отпечатка) на противоположное."""
    example = get_pattern(chat_id, pattern_id)
    if example is None:
        send_message("Этот пример уже удалён.", chat_id, {"inline_keyboard": [[TO_MENU]]})
        return
    swap = lambda k: (k.replace("bullish", "§").replace("bearish", "bullish").replace("§", "bearish")
                      if isinstance(k, str) else k)
    kind = swap(example.get("kind"))
    fp = dict(example.get("fingerprint") or {})
    if fp:
        fp["kind"] = swap(fp.get("kind"))
        fp["bullish"] = not fp.get("bullish", True)
    update_pattern(chat_id, pattern_id, kind=kind, fingerprint=fp or example.get("fingerprint"))
    word = "бычий" if str(kind).startswith("bullish") else "медвежий"
    edit_keyboard(chat_id, message_id, {"inline_keyboard": [
        [{"text": f"✅ Запомнил: {word}", "callback_data": f"ok:{pattern_id}"}],
        [{"text": "✏️ Описать словами", "callback_data": f"describe:{pattern_id}"}]]})
    send_message(f"Понял, это {word} «{example['name']}». Буду искать такие.", chat_id,
                 saved_keyboard(ALL_PATTERNS.index(example["name"])) if example["name"] in ALL_PATTERNS
                 else {"inline_keyboard": [[TO_MENU]]})


def understood_text(spec):
    """Как бот понял описание: строки «обязательно / желательно / нельзя …» и что не понял."""
    if not spec or language.is_empty(spec):
        return ""
    lines = ["📝 Понял твоё описание так:"]
    if spec.get("explain"):
        lines.append(f"🧠 {spec['explain']}")
    lines += [f"• {line}" for line in language.describe(spec)]
    if spec.get("unknown"):
        lines.append("🤔 Не понял: " + "; ".join(f"«{u}»" for u in spec["unknown"][:4])
                     + " — перефразируй, если это важно")
    return "\n".join(lines) + "\n"


def doubts_text(spec):
    """В чём бот не уверен — честно показываем и просим проверить."""
    lines = []
    if "direction" in (spec.get("doubts") or []):
        lines.append("⚠️ Направление взял по твоим словам — нейросеть поняла иначе. Проверь.")
    guessed = [language.CONCEPTS[k][0] for k in spec.get("guessed") or [] if k in language.CONCEPTS]
    if guessed:
        lines.append("🤔 Это я понял между строк — проверь: " + "; ".join(guessed[:4]))
    return "\n".join(lines) + ("\n" if lines else "")


def spec_keyboard(pattern_id, spec):
    """Под «понял так»: ✅ верно, 🔄 направление наоборот, ✖ убрать лишнее условие, ✏️ переписать.
    Так пользователь за секунду поправит бота, если тот понял не так."""
    rows = [[{"text": "✅ Всё верно", "callback_data": f"spok:{pattern_id}"}]]
    if spec.get("direction"):
        rows[0].append({"text": "🔄 Направление наоборот", "callback_data": f"spflip:{pattern_id}"})
    remove = [(k, language.CONCEPTS[k][0]) for k in spec.get("require", []) + spec.get("prefer", [])
              + spec.get("forbid", []) if k in language.CONCEPTS][:6]
    if spec.get("timeframes"):
        remove.append(("tf", "таймфреймы " + " ".join(t.upper() for t in spec["timeframes"])))
    buttons = [{"text": "✖ " + (label if len(label) <= 24 else label[:23] + "…"),
                "callback_data": f"sprm:{pattern_id}:{key}"} for key, label in remove]
    rows += [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([{"text": "✏️ Переписать", "callback_data": f"describe:{pattern_id}"},
                 {"text": "📂 Мои паттерны", "callback_data": "my"}])
    return {"inline_keyboard": rows}


def spec_message(name, spec, head):
    return (f"{head}\n\n{understood_text(spec)}{doubts_text(spec)}\n"
            "Всё верно? Если нет — убери лишнее кнопкой ✖ или поменяй направление.")


def same_description(chat_id, pattern_id):
    """Пример и все примеры того же паттерна с тем же описанием (описание общее для паттерна)."""
    example = get_pattern(chat_id, pattern_id)
    if example is None or not example.get("spec"):
        return None, []
    text = example["spec"].get("text")
    return example, [p for p in get_patterns(chat_id)
                     if p["name"] == example["name"] and (p.get("spec") or {}).get("text") == text]


def edit_spec(chat_id, message_id, pattern_id, action, key=None):
    """Кнопки под «понял так»: поправить описание у всех примеров с этим описанием."""
    example, group = same_description(chat_id, pattern_id)
    if example is None:
        send_message("Этот пример уже удалён.", chat_id, {"inline_keyboard": [[TO_MENU]]})
        return
    spec = dict(example["spec"])
    if action == "ok":
        edit_keyboard(chat_id, message_id, {"inline_keyboard": [
            [{"text": "✅ Запомнил", "callback_data": "noop"}],
            [{"text": "🔍 Сканировать", "callback_data": "scan"}, TO_MENU]]})
        return
    if action == "flip" and spec.get("direction"):
        spec["direction"] = "down" if spec["direction"] == "up" else "up"
        spec["doubts"] = [d for d in spec.get("doubts") or [] if d != "direction"]
    elif action == "rm" and key == "tf":
        spec["timeframes"] = []
    elif action == "rm" and key:
        for mode in ("require", "prefer", "forbid"):
            spec[mode] = [k for k in spec.get(mode, []) if k != key]
        spec["understood"] = [u for u in spec.get("understood", []) if (u[0] if isinstance(u, (list, tuple)) else u) != key]
        spec["guessed"] = [k for k in spec.get("guessed") or [] if k != key]
    for p in group:
        set_spec(chat_id, p["id"], spec)
    if language.is_empty(spec):
        edit_message(chat_id, message_id, f"Описание для «{example['name']}» убрал — ищу по скринам и оценкам.",
                     {"inline_keyboard": [[{"text": "✏️ Описать заново", "callback_data": f"describe:{pattern_id}"},
                                           TO_MENU]]})
        return
    edit_message(chat_id, message_id, spec_message(example["name"], spec, f"✏️ Поправил «{example['name']}»."),
                 spec_keyboard(pattern_id, spec))


def attach_spec(chat_id, pattern_id):
    """Описание из подписи к скрину — к сохранённому примеру. Возвращает текст «как понял»."""
    spec = pending_spec.get(chat_id)
    if not spec:
        return ""
    set_spec(chat_id, pattern_id, spec)
    return "\n" + understood_text(spec)


def receive_description(chat_id, pattern_id, text):
    """Пользователь описал паттерн словами — переводим в условия и запоминаем к примеру."""
    example = get_pattern(chat_id, pattern_id)
    if example is None:
        send_message("Этот пример уже удалён.", chat_id, {"inline_keyboard": [[TO_MENU]]})
        return
    smart = brain.available() and billing.is_pro(chat_id) and brain_allowed(chat_id)
    if smart:
        brain_notice(chat_id, "🧠 Думаю над описанием…")
    spec = language.understand(text, example["name"], smart=smart)
    keyboard = {"inline_keyboard": [
        [{"text": "✏️ Переписать", "callback_data": f"describe:{pattern_id}"},
         {"text": "📂 Мои паттерны", "callback_data": "my"}],
        [{"text": "🔍 Сканировать", "callback_data": "scan"}, TO_MENU]]}
    if language.is_empty(spec):
        send_message("Не нашёл в описании условий, которые умею проверять. "
                     + (f"Не понял: {'; '.join(spec['unknown'][:4])}.\n\n" if spec["unknown"] else "\n")
                     + DESCRIBE_TEXT, chat_id, keyboard)
        return
    set_spec(chat_id, pattern_id, spec)
    send_message(spec_message(example["name"], spec, f"✅ Запомнил для «{example['name']}»."),
                 chat_id, spec_keyboard(pattern_id, spec))


NOT_UNDERSTOOD = ("🤔 Не понял. Мне можно написать так:\n"
                  "• SOL — проверю монету\n"
                  "• скан — поищу паттерны на рынке\n"
                  "• что такое FVG — объясню\n"
                  "• бычий OB с объёмом на 4ч — запомню, каким должен быть твой паттерн")


def explain(chat_id, wish):
    """«Что такое FVG?», «объясни ликвидность» — короткий ответ сразу, без нейросети."""
    name, term = wish.get("pattern"), wish.get("term")
    if name:
        index = ALL_PATTERNS.index(name)
        opened = "gal" if examples_of(chat_id, index) else "pick"
        rows = [[{"text": f"🧩 Открыть «{name}»", "callback_data": f"{opened}:{index}" + (":0" if opened == "gal" else "")},
                 {"text": "👀 Пример с рынка", "callback_data": f"demo:{index}"}], [TO_MENU]]
        send_message(f"🧩 {name}\n\n{PATTERN_INFO[name]}", chat_id, {"inline_keyboard": rows})
    else:
        send_message(f"📖 {GLOSSARY[term]}", chat_id, {"inline_keyboard": [[TO_MENU]]})


def quick_reply(chat_id, wish):
    """Простая просьба, понятая словарём (language.quick_intent). True — если ответили."""
    kind = wish["intent"]
    if kind == "explain":
        explain(chat_id, wish)
    elif kind == "coin":
        find_coin(chat_id, wish["coin"])
    elif kind == "scan":
        start_job(chat_id, run_scan)
    elif kind == "history":
        start_job(chat_id, run_history)
    elif kind == "learning":
        show_view(chat_id, None, learning_view(chat_id))
    elif kind == "subscribe":
        show_view(chat_id, None, subscription_view(chat_id))
    elif kind == "profile":
        show_view(chat_id, None, profile_view(chat_id))
    elif kind == "settings":
        show_view(chat_id, None, settings_view(chat_id))
    elif kind == "patterns":
        show_view(chat_id, None, my_patterns_view(chat_id))
    elif kind == "help":
        send_welcome(chat_id)
    elif kind == "pattern":                          # просто «фвг» — открываем этот паттерн
        index = ALL_PATTERNS.index(wish["pattern"])
        if examples_of(chat_id, index):
            show_gallery(chat_id, index, 0)
        else:
            show_view(chat_id, None, intro_view(chat_id, index))
    elif kind in ("menu", "autoscan"):               # автоскан включается кнопкой в меню
        show_menu(chat_id)
    else:
        return False
    return True


def free_text(chat_id, text):
    """Свободное сообщение в чате. Бот понимает, чего хочет человек, в три шага:
    1) словарь: назван паттерн и есть условия → это описание («FVG на 4ч, свежий»);
    2) словарь: простая просьба → сразу делаем («скан», «покажи биткоин», «что такое фвг») —
       мгновенно и у всех, опечатки и чужая раскладка не мешают;
    3) нейросеть (brain.py, Pro) — для всего остального: вопросы, сложные формулировки.
    Никто не понял — подсказываем, как можно написать."""
    wish = None
    named = language.detect_pattern(text)
    quick = language.quick_intent(text)
    smart = brain.available() and billing.is_pro(chat_id)
    if quick and quick["intent"] == "explain" and not smart:
        explain(chat_id, quick)                      # с нейросетью на вопрос ответит она — подробнее
        return
    if named and not language.is_empty(language.parse(text, named)) and not (quick and quick["intent"] == "explain"):
        wish = {"intent": "describe", "pattern": named}       # «FVG на 4ч, свежий» — понятно и без нейросети
    elif quick and quick["intent"] != "explain" and quick_reply(chat_id, quick):
        return
    elif smart and brain_allowed(chat_id):
        brain_notice(chat_id, None)
        wish = brain.intent(language.fix_text(text), ALL_PATTERNS)
    if wish is None:
        if quick and quick_reply(chat_id, quick):    # нейросеть не ответила — объясняем словарём
            return
        send_message(NOT_UNDERSTOOD, chat_id, menu_keyboard(chat_id))
        return
    kind = wish.get("intent")
    if kind == "describe":
        describe_pattern(chat_id, wish.get("pattern") or named, text)
    elif kind == "question":
        reply = brain.answer(text)
        if reply:
            send_message("🧠 " + reply, chat_id, {"inline_keyboard": [[TO_MENU]]})
        elif not (quick and quick_reply(chat_id, quick)):
            show_menu(chat_id)
    elif kind == "scan":
        start_job(chat_id, run_scan)
    elif kind == "coin" and TICKER.match((wish.get("coin") or "").replace(" ", "")):
        find_coin(chat_id, wish["coin"].replace(" ", ""))
    elif kind == "learning":
        show_view(chat_id, None, learning_view(chat_id))
    elif quick and quick_reply(chat_id, quick):
        pass
    else:
        show_menu(chat_id)


BRAIN_PER_HOUR = 30                                # запросов к нейросети в час на человека
brain_calls = {}


def brain_allowed(chat_id, now=None):
    """Не больше BRAIN_PER_HOUR обращений к нейросети в час: один человек не займёт её для всех."""
    now = now or time.time()
    calls = [t for t in brain_calls.get(chat_id, []) if now - t < 3600]
    if len(calls) >= BRAIN_PER_HOUR:
        return False
    brain_calls[chat_id] = calls + [now]
    return True


def brain_notice(chat_id, text):
    """«Думаю…» — и честно про очередь, если нейросеть сейчас занята другим."""
    if brain.busy():
        send_message("🧠 Нейросеть сейчас отвечает другому — ты следующий, это займёт чуть дольше.", chat_id)
    elif text:
        send_message(text, chat_id)


def describe_pattern(chat_id, name, text):
    """Описание паттерна прямо в чате (без кнопки «✏️ Описать словами») — ко всем его примерам."""
    keyboard = {"inline_keyboard": [[{"text": "📂 Мои паттерны", "callback_data": "my"}, TO_MENU]]}
    if name is None:
        send_message("Понял, что ты описываешь паттерн, но не понял какой. Напиши название "
                     "(например «OB …», «FVG …», «канал …») или открой его в «📂 Мои паттерны» → «✏️ Описать словами».",
                     chat_id, keyboard)
        return
    examples = [p for p in get_patterns(chat_id) if p["name"] == name]
    if not examples:
        send_message(f"У тебя пока нет паттерна «{name}». Добавь его в «📂 Мои паттерны» "
                     "(можно без скриншота), потом пришли описание ещё раз.", chat_id, keyboard)
        return
    smart = brain.available() and billing.is_pro(chat_id) and brain_allowed(chat_id)
    if smart:
        brain_notice(chat_id, f"🧠 Думаю над описанием «{name}»…")
    spec = language.understand(text, name, smart=smart)
    if language.is_empty(spec):
        send_message(f"Понял, что это про «{name}», но не нашёл условий, которые умею проверять.\n\n"
                     + DESCRIBE_TEXT, chat_id, keyboard)
        return
    for example in examples:
        set_spec(chat_id, example["id"], spec)
    count = plural(len(examples), "пример", "примера", "примеров")
    send_message(spec_message(name, spec, f"✅ Запомнил для «{name}» ({count})."),
                 chat_id, spec_keyboard(examples[0]["id"], spec))


def receive_example(chat_id, name, path):
    """SMC, Range, флаг / вымпел: читаю свечи со скрина, нахожу на них твой паттерн
    тем же правилом, что и на рынке, и снимаю отпечаток — с ним буду сравнивать находки."""
    index = ALL_PATTERNS.index(name)
    drawn_box = None
    zones, marks = [], []
    if name == "Range":                                  # нарисованный прямоугольник = твой боковик
        candles, drawn_box = extract_range(path)
    else:
        candles, zones, marks = extract_marks(path)       # + зоны и линии, которые ты нарисовал
    if drawn_box is not None:
        fp, item = range_from_box(candles, drawn_box)
    elif candles is not None:
        fp, item = screenshot_fingerprint(name, candles, drawn_levels(path), zones, marks,
                                          spec=pending_spec.get(chat_id))
    else:
        fp, item = None, None
    if fp is None:
        remove_file(path)
        seen = f"вижу {len(candles)} свечей, но" if candles is not None else "не смог прочитать свечи, и"
        send_message(f"На скрине {seen} не нашёл «{name}» по правилу:\n{PATTERN_INFO[name]}\n\n"
                     "Для сравнения нужен другой скрин: "
                     "свечи крупно, видно 20–30 свечей до паттерна и несколько после, без индикаторов "
                     "(объём снизу — можно, я его прочитаю).", chat_id,
                     retry_keyboard(index))
        return
    pattern_id = add_pattern(chat_id, name, path, note="скрин", kind=item.get("kind"), fingerprint=fp)
    understood = attach_spec(chat_id, pattern_id)
    debug_path = f"{PATTERNS_DIR}/{chat_id}_debug.png"
    with chart_lock:
        plot_example(candles, "", "", item, debug_path, f"Что я увидел на твоём скрине: {name}", dates=False)
    drawn = ("• нашёл твой прямоугольник" + (" и линию POC" if drawn_box.get("poc") is not None else "")
             + " — границы боковика беру с него\n") if drawn_box is not None else (
        "• нашёл закрашенную тобой зону — паттерн беру по ней\n" if zones else
        "• нашёл твою разметку линиями — паттерн беру по ней\n" if item and item.get("from_line") else "")
    caption = (f"Вот что я увидел на твоём скрине:\n• {kind_text(item)}\n{drawn}"
               f"{understood}\n"
               "Буду искать похожие. Я правильно нашёл паттерн?")
    keyboard = example_keyboard(pattern_id, item)
    send_photo(debug_path, caption[:CAPTION_LIMIT], chat_id, keyboard)
    remove_file(debug_path)


def receive_shape(chat_id, name, path):
    """Двойное дно / вершина, ГиП: читаю свечи со скрина и нахожу фигуру по ТОЧКАМ (два дна и шея,
    голова и плечи) — тем же правилом, что и на рынке. Эталон формы — только сама фигура, без
    лишнего графика вокруг. False — по точкам не нашёл (тогда работаем по форме всего скрина)."""
    from shape import normalize, resample
    from shapes_check import read_shape, shape_label
    from templates import SHAPE_POINTS
    found = read_shape(name, path)
    if found is None:
        return False
    candles, figure = found
    xs = [x for x, _ in figure["points"]]
    segment = candles["close"].to_numpy()[xs[0]:xs[-1] + 1]
    if len(segment) < 8:
        return False
    shape = [float(v) for v in normalize(resample(segment, SHAPE_POINTS))]
    pattern_id = add_pattern(chat_id, name, path, note="скрин: фигура по точкам", shape=shape)
    understood = attach_spec(chat_id, pattern_id)
    item = {**figure, "name": name, "start": xs[0], "window": xs[-1] - xs[0] + 1}
    debug_path = f"{PATTERNS_DIR}/{chat_id}_debug.png"
    with chart_lock:
        plot_example(candles, "", "", item, debug_path, f"Что я увидел на твоём скрине: {name}", dates=False)
    caption = (f"Вот что я увидел на твоём скрине ({len(candles)} свечей):\n"
               f"• {shape_label(name, figure).split(':')[0]} — точки фигуры отмечены ломаной, шея — пунктиром\n"
               f"• {'шея пробита — фигура подтверждена' if figure.get('confirmed') else 'шея ещё не пробита'}\n\n"
               f"{understood}\n"
               "На рынке буду искать фигуры такой же формы. Я правильно нашёл паттерн?")
    keyboard = example_keyboard(pattern_id)
    send_photo(debug_path, caption[:CAPTION_LIMIT], chat_id, keyboard)
    remove_file(debug_path)
    return True


def receive_pattern_photo(chat_id, name, file_id, caption=""):
    path = f"{PATTERNS_DIR}/{chat_id}_{int(time.time() * 1000)}.jpg"
    download_photo(file_id, path)
    try:
        _read_pattern_photo(chat_id, name, path, caption)
    except UnreliableScreenshot as error:
        remove_file(path)
        send_message(str(error), chat_id, retry_keyboard(ALL_PATTERNS.index(name)))


def _read_pattern_photo(chat_id, name, path, caption):
    index = ALL_PATTERNS.index(name)
    # подпись к скрину — описание паттерна словами: бот учитывает его, когда ищет паттерн на скрине
    spec = language.understand(caption, name, smart=billing.is_pro(chat_id)) if (caption or "").strip() else None
    pending_spec[chat_id] = spec if spec and not language.is_empty(spec) else None

    if name in FAMILY:                                    # SMC, Range, флаг, вымпел — по отпечатку
        receive_example(chat_id, name, path)
        return
    if name in SHAPE_PATTERNS and receive_shape(chat_id, name, path):
        return                                            # нашёл фигуру по точкам на свечах скрина

    analysis = analyze_screenshot(path)
    if not analysis.ok:
        remove_file(path)
        send_message("Не смог разобрать график на картинке.\n\n" + TIPS, chat_id, retry_keyboard(index))
        return

    lines_text = describe_lines(analysis.lines)
    figure = kind_from_drawing(analysis.lines)
    if figure is None and name in LINE_PATTERNS:          # линий не нарисовал — ищу фигуру по свечам
        from history import read_lines_figure
        seen = read_lines_figure(name, path)
        if seen is not None:
            figure = seen[1]["kind"]
            lines_text = "линии фигуры нашёл сам по свечам"
    if name in LINE_PATTERNS and figure not in FAMILY_BY_PATTERN[name]:
        reason = f"По границам вижу «{KIND_NAMES[figure]}»." if figure else "Не удалось выделить две границы фигуры."
        raise UnreliableScreenshot(f"{reason} Для «{name}» отметь обе стороны или выбери другой тип паттерна.")
    pattern_id = add_pattern(chat_id, name, path, lines_text, figure)
    understood = attach_spec(chat_id, pattern_id)

    debug_path = f"{PATTERNS_DIR}/{chat_id}_debug.png"
    draw_debug(path, analysis, debug_path)
    chart_kind = "свечной график" if analysis.mode == "candles" else "линию цены"
    caption = f"Вот что я увидел в «{name}»:\n• {chart_kind} — жёлтым\n• {lines_text}\n"
    if figure:
        caption += f"• Фигура: {KIND_NAMES[figure]}\n"
    caption += f"{understood}\nВсё верно?"
    keyboard = example_keyboard(pattern_id)
    send_photo(debug_path, caption, chat_id, keyboard)
    remove_file(debug_path)


TICKER = re.compile(r"^[A-Za-z0-9]{2,15}(/?USDT)?$", re.IGNORECASE)


WELCOME_KEYBOARD = {"inline_keyboard": [
    [GUIDE_BUTTON], *menu_keyboard()["inline_keyboard"]]}

photo_queries = OrderedDict()
photo_lock = threading.Lock()


def leave_photo_flow(chat_id):
    waiting.pop(chat_id, None)
    describing.pop(chat_id, None)
    with photo_lock:
        photo_queries.pop(chat_id, None)


def show_tutorial(chat_id, message_id, media, page):
    _, filename, caption = SCREENSHOT_GUIDE[page]
    path = os.path.join(os.path.dirname(__file__), "tests", "samples", filename)
    tabs = [{"text": ("• " if index == page else "") + label, "callback_data": f"tutorial:{index}"}
            for index, (label, _, _) in enumerate(SCREENSHOT_GUIDE)]
    keyboard = {"inline_keyboard": [tabs, [{"text": "🔍 Поиск", "callback_data": "search"}, TO_MENU]]}
    if media and edit_photo(chat_id, message_id, path, caption, keyboard):
        return
    sent = send_photo(path, caption, chat_id, keyboard)
    if sent and message_id:
        delete_message(chat_id, message_id)


def inspect_chat_photo(chat_id, file_id):
    token = secrets.token_hex(6)
    with photo_lock:
        photo_queries[chat_id] = (token, time.monotonic() + 1800, [])
        photo_queries.move_to_end(chat_id)
        if len(photo_queries) > 512:
            photo_queries.popitem(last=False)
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "chart.jpg")
            download_photo(file_id, path)
            choices = screenshot_choices(path)
    except UnreliableScreenshot as error:
        choices, reason = [], str(error)
    except (ValueError, OSError) as error:
        if isinstance(error, requests.RequestException):
            raise
        choices, reason = [], "Не удалось прочитать изображение. Пришли фото свечного графика."
    with photo_lock:
        current = photo_queries.get(chat_id)
        if not current or current[0] != token:
            return
        if choices:
            photo_queries[chat_id] = (token, current[1], choices)
        else:
            del photo_queries[chat_id]
    if not choices:
        send_message(reason, chat_id, {"inline_keyboard": [[GUIDE_BUTTON], [TO_MENU]]})
        return
    single = len(choices) == 1
    text = f"Вижу {choices[0][0]}. Искать похожие графики?" if single else "Какой из отмеченных паттернов искать?"
    rows = [[{"text": "🔍 Искать похожие" if single else label, "callback_data": f"query:{token}:{i}"}]
            for i, (label, _) in enumerate(choices)]
    send_message(text, chat_id, {"inline_keyboard": rows + [[GUIDE_BUTTON, TO_MENU]]})


def confirm_chat_photo(chat_id, token, index, message_id):
    with photo_lock:
        query = photo_queries.get(chat_id)
        valid = query and query[0] == token and query[1] > time.monotonic() and 0 <= index < len(query[2])
        if valid:
            _, example = query[2][index]
            del photo_queries[chat_id]
    if not valid:
        show_view(chat_id, message_id, ("Эта кнопка устарела. Пришли скриншот для нового поиска.",
                                        {"inline_keyboard": [[GUIDE_BUTTON], [TO_MENU]]}))
        return
    run_scan(chat_id, example, message_id)


def send_welcome(chat_id):
    st = billing.status(chat_id)
    gift = ""
    if st["trial"] and st["pro"]:
        gift = (f"\n\nПервые {billing.TRIAL_DAYS} дня — Pro бесплатно (до {billing.until_text(st['until'])}): "
                "все паттерны, автоскан, нейросеть. Потом Free или подписка.")
    send_message(WELCOME + gift, chat_id, WELCOME_KEYBOARD)


def handle_command(chat_id, text):
    """Команды из меню «/» Telegram. Возвращает True, если команда узнана."""
    command, _, argument = text.partition(" ")
    command = command.split("@")[0].lower()             # /scan@findercandles_bot — тоже /scan
    if admin_command(chat_id, command, argument):
        return True
    if command in ("/subscribe", "/plan"):
        show_view(chat_id, None, subscription_view(chat_id))
    elif command in ("/profile", "/me"):
        leave_photo_flow(chat_id)
        show_view(chat_id, None, profile_view(chat_id))
    elif command == "/start":
        leave_photo_flow(chat_id)
        send_welcome(chat_id)
    elif command == "/help":
        leave_photo_flow(chat_id)
        show_view(chat_id, None, (HELP_TEXT, WELCOME_KEYBOARD))
    elif command == "/menu":
        show_menu(chat_id)
    elif command == "/patterns":
        leave_photo_flow(chat_id)
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
    if "successful_payment" in message:
        handle_payment(message)
        return
    if "refunded_payment" in message:
        handle_refunded(message)
        return
    remember_username(message)
    chat_id = message["chat"]["id"]
    if chat_id in writing_broadcast and message.get("text") and not message["text"].startswith("/"):
        writing_broadcast.discard(chat_id)
        draft_broadcast(chat_id, message["text"])
        return
    text = message.get("text", "").strip()
    if first_visit(chat_id) and not text.startswith(("/start", "/help")):
        send_welcome(chat_id)                          # новичку — сначала объяснение

    if "photo" not in message:
        if chat_id in describing and text and not text.startswith("/"):
            background(receive_description, chat_id, describing.pop(chat_id), text)
            return
        if text.startswith("/"):
            # несколько команд в одном сообщении (каждая с новой строки) — выполняем все
            commands = [line.strip() for line in text.splitlines() if line.strip().startswith("/")]
            if len(commands) > 1 and billing.is_admin(chat_id):
                for line in commands:
                    handle_command(chat_id, line)
                return
            if not handle_command(chat_id, text):
                show_menu(chat_id)
            return
        ticker = text.replace(" ", "")
        if TICKER.match(ticker):
            find_coin(chat_id, ticker)           # написали тикер — ищем монету
            return
        if text:
            background(free_text, chat_id, text)  # любой текст: описание, вопрос, «просканируй», монета
            return
        show_menu(chat_id)
        return

    if chat_id not in waiting:
        describing.pop(chat_id, None)
        start_job(chat_id, inspect_chat_photo, message["photo"][-1]["file_id"])
        return

    name = waiting.pop(chat_id)
    file_id = message["photo"][-1]["file_id"]
    try:
        receive_pattern_photo(chat_id, name, file_id, message.get("caption", ""))
    finally:
        pending_spec.pop(chat_id, None)


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


def record_vote(chat_id, coin, index, vote, quiet=False):
    """Запоминает оценку находки, сразу переучивается и рассказывает, ЧТО изменилось.
    Возвращает текст ответа (его же и присылает, если quiet=False)."""
    tf, item, df = coin_entries(coin)[index]
    name = item["name"]
    before = learn(get_feedback(chat_id)).get(name, {"records": [], "rules": []})
    if item.get("factors") is None:
        strength.evaluate(df, item)
    direction = ("up" if str(item.get("kind", "")).startswith("bullish") else
                 "down" if str(item.get("kind", "")).startswith("bearish") else None)
    record = {"key": feedback_key(coin["symbol"], tf, item, df), "at": time.time(),
              "symbol": coin["symbol"], "interval": tf, "name": name, "vote": vote,
              "features": features(df, item), "window": item_window(df, item),
              "factors": item.get("factors"), "group": item.get("group"),
              "facts": language.present(df, item), "direction": direction, "kind": item.get("kind")}
    if vote > 0:                                                      # 👍 — ещё один пример
        try:
            if name in FAMILY:
                record["fingerprint"] = make_fingerprint(name, df, item)
            else:                                                     # фигура: эталон линий / формы
                example = market_example(df, item)
                if example and (example.get("reference") or example.get("shape")):
                    record["example"] = {k: v for k, v in example.items() if v is not None}
        except Exception:
            pass
    after = learn(add_feedback(chat_id, record)).get(name)
    if vote > 0:
        head = (f"👍 «{name}»: запомнил как ещё один пример — теперь ищу и похожие на эту находку."
                if record.get("fingerprint") or record.get("example") else f"👍 «{name}»: запомнил, такое тебе подходит.")
    else:
        head = f"👎 «{name}»: эту находку больше не пришлю, похожие на неё опущу ниже."
    lines = [head]
    changes = what_changed(before, after, vote)
    if changes:
        lines.append("🎓 Переучился по твоим оценкам:\n• " + "\n• ".join(changes))
    if after["rules"] and after["rules"] != before["rules"]:
        lines.append("Присылаю только: " + "; ".join(rule_text(r) for r in after["rules"]))
    if after.get("head") and not before.get("head"):
        lines.append("Запомнил, как выглядят находки, которые тебе нравятся, — отбор учитывает и вид.")
    if not changes and len(after["records"]) < 6:
        likes = sum(r["vote"] > 0 for r in after["records"])
        lines.append(f"Оценок по «{name}»: 👍 {likes}, 👎 {len(after['records']) - likes}. "
                     "С каждой следующей уточняю, что для тебя важно (📚 Обучение).")
    text = "\n".join(lines)
    if not quiet:
        send_message(text, chat_id)
    return text


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
        name = finding_name(item)
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
        text = record_vote(chat_id, coin, 0, vote, quiet=True)
        if message_id is not None:                       # просмотрщик: ✅ на кнопке
            edit_keyboard(chat_id, message_id, card_keyboard(card_id, coin, chat_id))
        keyboard = None
        if vote > 0 and "пример" not in text and market_example(entries[0][2], entries[0][1]) is not None:
            keyboard = {"inline_keyboard": [[{"text": "💾 Сохранить как пример", "callback_data": f"ex:{card_id}:0"}]]}
        send_message(text, chat_id, keyboard)
        return
    send_message(f"{coin['symbol']}: оцени каждую находку отдельно —\n"
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
    if not billing.can_add_pattern(chat_id, item["name"]):
        paywall(chat_id, "patterns")
        return
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
    record_vote(chat_id, coin, index, 1, quiet=True)            # сохранил — значит, нравится
    index_name = ALL_PATTERNS.index(item["name"]) if item["name"] in ALL_PATTERNS else None
    keyboard = {"inline_keyboard": [[CLOSE]]}
    if index_name is not None:
        keyboard = {"inline_keyboard": [[{"text": "🖼 Посмотреть примеры", "callback_data": f"gal:{index_name}:{count - 1}"},
                                         CLOSE]]}
    send_message(f"✅ Сохранил как пример №{count} к «{item['name']}». Теперь ищу и похожие на него.",
                 chat_id, keyboard)


def learning_view(chat_id, note=""):
    keyboard = {"inline_keyboard": [[{"text": "♻️ Забыть все оценки", "callback_data": "fbreset"}],
                                    [TO_MENU]]}
    return note + learning_summary(get_feedback(chat_id)), keyboard


def search_patterns(chat_id, patterns=None):
    """Примеры пользователя + выученное по 👍/👎 (условия и понравившиеся находки как примеры)."""
    patterns = get_patterns(chat_id) if patterns is None else patterns
    patterns = billing.allowed_patterns(chat_id, patterns)        # Free: только 2 паттерна
    if not get_settings(chat_id).get("learn", True):
        return patterns
    return with_learning(patterns, get_feedback(chat_id))


def user_judge(chat_id, settings):
    """Чему бот научился по оценкам этого человека (или None, если обучение выключено)."""
    return make_judge(get_feedback(chat_id)) if settings.get("learn", True) else None


# --- кнопка «🧩 Паттерны»: выбрать, что нанести на график ---------------------------------------
selections = {}          # (chat_id, card_id) → {"off": номера снятых галочек, "history": bool}


def coin_items(coin):
    """Все находки монеты по порядку: [(таймфрейм, находка)]."""
    return [(f["interval"], it) for f in coin["frames"] if not f["history"] for it in f["items"]]


def selection(chat_id, card_id):
    return selections.setdefault((chat_id, card_id), {"off": set(), "history": True})


def patterns_keyboard(chat_id, card_id, coin):
    sel = selection(chat_id, card_id)
    rows = []
    for i, (tf, item) in enumerate(coin_items(coin)):
        name = finding_name(item)
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
        send_message("Ничего не выбрано — отметь хотя бы один паттерн.", chat_id)
        return
    send_action(chat_id)                         # «отправляет файл…» вверху чата
    patterns = [coin["query_example"]] if "query_example" in coin else search_patterns(chat_id)
    groups = selected_markup(patterns, coin, chosen, get_settings(chat_id))
    code = pine_script(coin["symbol"], chosen[0][0], groups, history_default=sel["history"])
    send_markup(chat_id, coin, groups, code, f"{coin['symbol']}_selected.pine.txt", reply_to,
                selected=True, history_default=sel["history"])


def send_pine(chat_id, card_id, reply_to=None):
    coin = find_card(chat_id, card_id)
    if coin is None:
        outdated(chat_id)
        return
    send_action(chat_id)                         # «отправляет файл…» вверху чата
    patterns = [coin["query_example"]] if "query_example" in coin else search_patterns(chat_id)
    groups = full_markup(patterns, coin, get_intervals(chat_id), get_settings(chat_id))
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
        names = ", ".join(dict.fromkeys(short_item(it) for it in f["items"]))
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
                     {"text": "Далее ▶️", "callback_data": f"v:{ids[(i + 1) % n]}"}])
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
    rows.append(bottom + [{"text": "🔍 Новый поиск", "callback_data": "search"}, HOME])
    return {"inline_keyboard": rows}


def short_item(item):
    """Находка коротко — только элемент анализа: «FVG ↓», «OB ↑», «Канал».
    Касания, похожесть на пример, цены зон — в «📄 Подробно»."""
    return short_name(item)


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
        lines.append(f"{f['interval'].upper()}: " + ", ".join(dict.fromkeys(short_item(it) for it in f["items"])))
    matches = [(f["interval"], item) for f in coin["frames"] for item in f["items"]]
    if matches:
        tf, item = max(matches, key=lambda pair: (pair[1].get("similarity") or 0, pair[1].get("score", 0)))
        score = item.get("similarity")
        if score is not None:
            lines.append(f"Сходство с примером · {tf.upper()} · {score:.0%}")
        reasons = strength.strongest(item, 2)
        if not reasons and item.get("name") == "FVG":
            reasons = ["между тенями первой и третьей свечи есть разрыв"]
        elif not reasons and item.get("upper") is not None and item.get("lower") is not None:
            reasons = [KIND_NAMES.get(item.get("kind"), "две границы фигуры")]
            if isinstance(item.get("touches"), int):
                reasons.append(plural(item["touches"], "касание", "касания", "касаний"))
        elif not reasons and item.get("points"):
            reasons = ["экстремумы образуют фигуру «" + item["name"] + "»"]
        if not reasons and item.get("name") in PATTERN_INFO:
            reasons = [PATTERN_INFO[item["name"]].split(". ")[0].rstrip(".")]
        if reasons:
            lines.append(f"Почему ({short_item(item)}, {tf.upper()}): " + ", ".join(reasons))
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
FIND_TEXT = "🔎 Напиши тикер монеты, например SOL или PEPE."


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
        send_message(f"Не нашёл {symbol} на Binance. Проверь тикер (нужна пара к USDT).", chat_id,
                     {"inline_keyboard": [[TO_MENU]]})
        return
    tfs = ", ".join(i.upper() for i in intervals)
    wait_id = send_message(f"🔎 В последних результатах {symbol} нет — проверяю сейчас ({tfs})…", chat_id)
    # при поиске конкретной монеты показываем всё, что на ней есть: без «только комбинации»
    settings = {**get_settings(chat_id), "combos_only": False}
    cards, _ = scan_user(search_patterns(chat_id, patterns), intervals, symbols=[symbol], settings=settings,
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


def run_scan(chat_id, example=None, edit_id=None):
    patterns = get_patterns(chat_id)
    if example:
        allowed = billing.allowed_patterns(chat_id, [*patterns, example])
        if not any(p is example for p in allowed):
            paywall(chat_id, "patterns")
            return
        patterns = [example]
    if not patterns:
        send_message(NO_PATTERNS[0], chat_id, NO_PATTERNS[1])
        return
    if not billing.use_scan(chat_id):
        paywall(chat_id, "scans")
        return
    intervals = get_intervals(chat_id)
    tfs = ", ".join(i.upper() for i in intervals)
    coins = get_coins(chat_id)
    target = f" · {example['name']}" if example else ""
    wait_id = say(chat_id, edit_id, f"🔍 Сканирую топ-{coins} монет ({tfs}){target}, {wait_text(coins, intervals)}…")
    settings = get_settings(chat_id)
    if example:
        settings = {**settings, "similar_only": True, "combos_only": False}
    else:
        patterns = search_patterns(chat_id, patterns)
    cards, skipped = scan_user(patterns, intervals, symbols=top_symbols(coins), settings=settings,
                               judge=user_judge(chat_id, settings))
    if example:
        # Один эталон даёт не больше одной находки на монету и таймфрейм.
        matches = []
        for card in cards:
            item = card["items"][0]
            if any(example.get(key) and item.get(key) != example[key] for key in ("kind", "direction")):
                continue
            if example.get("shape"):
                item["similarity"] = item["score"]  # find_shape возвращает корреляцию с этим эталоном
            if (item.get("similarity") or 0) >= SIMILAR_ENOUGH:
                matches.append(card)
        cards = matches
    report_skipped(chat_id, skipped)
    again = {"inline_keyboard": [[{"text": "🔍 Новый поиск" if example else "🔍 Ещё раз",
                                  "callback_data": "search" if example else "scan"},
                                  {"text": "⚙️ Настройки", "callback_data": "settings"}], [TO_MENU]]}
    if not cards:
        text = "Совпадений сейчас нет."
        if settings["combos_only"]:
            text += " Включено «только комбинации» — одиночные находки не показываю (⚙️ Настройки)."
        say(chat_id, wait_id, text, again)
        return
    coins_found = group_by_coin(cards)
    if example:
        for coin in coins_found:
            coin["query_example"] = example
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
    if not billing.use_scan(chat_id):
        paywall(chat_id, "scans")
        return
    intervals = get_intervals(chat_id)
    coins = get_coins(chat_id)
    wait_id = send_message(f"📊 Считаю историю по топ-{coins} монет, {wait_text(coins * 2, intervals)}…", chat_id)
    reports, cards, skipped = history_user(search_patterns(chat_id, patterns), intervals, symbols=top_symbols(coins),
                                           settings=get_settings(chat_id))
    report_skipped(chat_id, skipped)
    say(chat_id, wait_id, f"📊 История по топ-{coins} монет готова.")
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
            report_error("Ошибка в фоне", error)
    views.submit(run)


def start_describing(chat_id, pattern_id):
    describing[chat_id] = pattern_id
    example = get_pattern(chat_id, pattern_id)
    current = understood_text(example.get("spec")) if example else ""
    send_message(DESCRIBE_TEXT + (BRAIN_TIP if brain.available() else "")
                 + (f"\n\nСейчас:\n{current}" if current else ""), chat_id,
                 {"inline_keyboard": [[{"text": "📂 Мои паттерны", "callback_data": "my"}, TO_MENU]]})


def handle_callback(query):
    message = query["message"]
    chat_id = message["chat"]["id"]
    message_id = message["message_id"]
    media = is_media(message)
    data = query["data"]
    answer_callback(query["id"])

    if data == "noop":
        return

    if data == "search":
        leave_photo_flow(chat_id)
        show_view(chat_id, message_id, SEARCH_VIEW, media)
        return

    if data.startswith("tutorial:"):
        page = data.partition(":")[2]
        if page.isdecimal() and int(page) < len(SCREENSHOT_GUIDE):
            show_tutorial(chat_id, message_id, media, int(page))
        return

    if data.startswith("query:"):
        parts = data.split(":")
        if len(parts) == 3 and parts[2].isdecimal():
            start_job(chat_id, confirm_chat_photo, parts[1], int(parts[2]), message_id)
        return

    if data.startswith("adm:"):
        admin_callback(chat_id, message_id, data[4:], media)
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
        leave_photo_flow(chat_id)
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

    if data == "sub":
        show_view(chat_id, message_id, subscription_view(chat_id), media)
        return

    if data == "profile":
        leave_photo_flow(chat_id)
        show_view(chat_id, message_id, profile_view(chat_id), media)
        return

    if data == "paycheck":
        background(check_card_payments, chat_id)
        return

    if data == "auto_section":
        leave_photo_flow(chat_id)
        show_view(chat_id, message_id, autoscan_view(chat_id), media)
        return

    if data == "auto":
        if not get_autoscan(chat_id) and not billing.is_pro(chat_id):
            paywall(chat_id, "autoscan")
            return
        toggle_autoscan(chat_id)
        show_view(chat_id, message_id, autoscan_view(chat_id), media)
        return

    if data == "help":
        leave_photo_flow(chat_id)
        show_view(chat_id, message_id, (HELP_TEXT, WELCOME_KEYBOARD), media)
        return

    if data == "sethelp":
        send_message(SETTINGS_HELP, chat_id, {"inline_keyboard": [[CLOSE]]})
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
            send_message("Для комбинаций нужно минимум 2 разных паттерна — сейчас находок не будет. "
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

    if action == "pay":
        background(pay_card, chat_id, value)
        return

    if action in ("on", "shot", "dsave") and not billing.can_add_pattern(chat_id, ALL_PATTERNS[int(value)]):
        paywall(chat_id, "patterns")
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

    if action == "describe":                   # описать пример словами
        start_describing(chat_id, value)
        return

    if action == "flipdir":                     # бот понял направление примера наоборот
        flip_direction(chat_id, message_id, value)
        return

    if action in ("spok", "spflip", "sprm"):    # поправить, как бот понял описание
        pattern_id, _, key = value.partition(":")
        edit_spec(chat_id, message_id, pattern_id, {"spok": "ok", "spflip": "flip", "sprm": "rm"}[action], key or None)
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
        report_error(f"Ошибка автоскана {intervals}", error)


class Chores:
    """Служебные дела по таймеру (в фоне, не мешая ответам):
    раз в минуту — проверить оплаты картой; раз в час — напоминания о продлении;
    раз в сутки — резервная копия базы."""

    EVERY = {"payments": 60, "reminders": 3600, "backup": 86400}

    def __init__(self):
        self.last = {name: 0 for name in self.EVERY}

    def run(self, now):
        for name, every in self.EVERY.items():
            if now - self.last[name] >= every:
                self.last[name] = now
                views.submit(self.safe, getattr(self, name))

    @staticmethod
    def safe(task):
        try:
            task()
        except Exception as error:
            report_error("Ошибка служебной задачи", error)

    @staticmethod
    def payments():
        check_card_payments()

    @staticmethod
    def reminders():
        for chat_id, text in billing.reminders():
            send_message(text, chat_id, {"inline_keyboard": [[{"text": "💎 Подписка", "callback_data": "sub"}]]})

    @staticmethod
    def backup():
        path = storage.backup()
        if path:
            print("💾 Резервная копия базы:", path)


def check_schedule(schedule):
    """Если закрылись свечи (в 00:00 UTC закрываются сразу 1H, 4H, 12H и 1D) —
    один автоскан на все эти ТФ, чтобы монета пришла одним сообщением."""
    due = schedule.due(time.time())
    if due:
        autoscan_jobs.submit(autoscan_job, due)


def network_wait(failures):
    """Сколько ждать после failures неудачных попыток подряд: 5, 10, 20, 40, 60, 60… секунд."""
    return min(60, 5 * 2 ** (failures - 1))


def network_hint(error, failures, wait):
    """Короткое сообщение об ошибке сети — без токена и с подсказкой, что делать."""
    kind = error.__class__.__name__
    text = f"Нет связи с Telegram ({kind}), попытка {failures}, жду {wait} с."
    if failures == 1:
        text += ("\n   Бот сам переподключится. Если это надолго: проверь интернет / VPN; "
                 "если api.telegram.org заблокирован — впиши прокси в .env: TELEGRAM_PROXY=http://адрес:порт")
        text += "\n   Подробно: " + safe_error(error)[:300]
    return text


def main():
    offset = 0
    schedule = Schedule(time.time())
    print("Глаз для сравнения «на вид»:", visual.load_config())   # нейросеть, если setup_ai.py её включил
    on = quality.load()                                            # ИИ-значимость, если setup_ai.py её обучил
    print("ИИ-значимость:", ", ".join(quality.FAMILY_TEXT[f] for f in on) or "нет (работают правила)")
    print("Мозг (понимание слов):", brain.status())
    brain.warm_up()
    print("Оплата: ⭐ Stars" + (" + 💳 ЮKassa" if billing.yookassa_on() else " (ЮKassa не настроена)")
          + f" · админ: {', '.join(map(str, billing.admins())) or 'не задан (ADMIN_IDS в .env)'}")
    try:
        set_commands(COMMANDS)                           # меню «/» в Telegram
    except requests.RequestException as error:
        print("Не смог обновить команды бота:", safe_error(error))
    print(f"Бот запущен (обновление {VERSION}). Остановить: Ctrl+C"
          + (" · через прокси" if PROXY else ""))
    failures = 0
    chores = Chores()
    while True:
        check_schedule(schedule)
        chores.run(time.time())
        try:
            updates = get_updates(offset)
        except requests.RequestException as error:
            failures += 1
            wait = network_wait(failures)
            if failures == 1 or failures % 10 == 0:          # не засыпаем консоль одинаковыми ошибками
                print(network_hint(error, failures, wait))
            time.sleep(wait)
            continue
        if failures:
            print(f"✅ Связь с Telegram восстановлена (было {failures} неудачных попыток)")
            failures = 0

        for update in updates:
            offset = update["update_id"] + 1
            try:
                if "message" in update:
                    handle_message(update["message"])
                elif "callback_query" in update:
                    handle_callback(update["callback_query"])
                elif "pre_checkout_query" in update:
                    handle_pre_checkout(update["pre_checkout_query"])
            except Exception as error:
                report_error("Ошибка при обработке", error)


if __name__ == "__main__":
    main()
