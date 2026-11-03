"""Память бота: паттерны и настройки пользователей в users.json.

Бот работает в несколько потоков (сканирование идёт в фоне), поэтому каждое изменение
файла — «прочитать → поменять → записать» — делается под замком LOCK: иначе два потока
прочитают старую версию, и изменения одного из них потеряются."""

import functools
import json
import os
import threading
import time

FILE = "users.json"
LOCK = threading.RLock()     # RLock: функция под замком может вызвать другую такую же


def locked(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with LOCK:
            return func(*args, **kwargs)
    return wrapper


@locked
def load_users():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    # старые записи без id: используем путь к фото как id
    for user in users.values():
        for p in user.get("patterns", []):
            p.setdefault("id", p["photo"])
            p.setdefault("note", "")
    return users


@locked
def save_users(users):
    # пишем во временный файл и подменяем: если программа упадёт посреди записи,
    # users.json останется целым
    tmp = FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FILE)


@locked
def add_pattern(chat_id, name, photo_path, note="", kind=None, reference=None, shape=None, builtin=False,
                fingerprint=None):
    """Сохраняет пример паттерна и возвращает его id. Примеров одного паттерна может быть много.
    kind — тип фигуры с рисунка пользователя (например, "symmetric"), если линии нарисованы;
    reference / shape — эталон, снятый прямо с рынка (кнопка «💾 Сохранить как пример»);
    builtin — поиск включён без скриншота (по правилам или встроенному эталону), photo_path = None;
    fingerprint — отпечаток примера SMC / Range / флага (fingerprint.py): с чем сравнивать находки."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    pattern_id = str(int(time.time() * 1000))
    while any(p["id"] == pattern_id for p in user["patterns"]):   # два сохранения в одну миллисекунду
        pattern_id = str(int(pattern_id) + 1)
    entry = {"id": pattern_id, "name": name, "photo": photo_path, "note": note, "kind": kind}
    if reference is not None:
        entry["reference"] = reference
    if shape is not None:
        entry["shape"] = shape
    if builtin:
        entry["builtin"] = True
    if fingerprint is not None:
        entry["fingerprint"] = fingerprint
    user["patterns"].append(entry)
    save_users(users)
    return pattern_id


def get_patterns(chat_id):
    return load_users().get(str(chat_id), {}).get("patterns", [])


@locked
def delete_pattern(chat_id, pattern_id):
    """Удаляет паттерн по id и возвращает его (или None, если не нашёлся)."""
    users = load_users()
    patterns = users.get(str(chat_id), {}).get("patterns", [])
    for i, p in enumerate(patterns):
        if p["id"] == pattern_id:
            removed = patterns.pop(i)
            save_users(users)
            return removed
    return None


DEFAULT_INTERVALS = ["4h"]


def get_intervals(chat_id):
    """Таймфреймы, выбранные пользователем (по умолчанию 4h)."""
    return load_users().get(str(chat_id), {}).get("intervals", DEFAULT_INTERVALS)


@locked
def toggle_interval(chat_id, interval, allowed):
    """Включает/выключает таймфрейм. Последний выключить нельзя. Возвращает новый список."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    current = user.get("intervals", list(DEFAULT_INTERVALS))
    if interval in current:
        if len(current) > 1:
            current.remove(interval)
    else:
        current.append(interval)
    user["intervals"] = [i for i in allowed if i in current]   # порядок как в меню
    save_users(users)
    return user["intervals"]


# --- автоскан ---------------------------------------------------------------------------------
SENT_KEEP_DAYS = 30
SENT_MAX = 1000


def get_autoscan(chat_id):
    return load_users().get(str(chat_id), {}).get("autoscan", False)


@locked
def toggle_autoscan(chat_id):
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    user["autoscan"] = not user.get("autoscan", False)
    save_users(users)
    return user["autoscan"]


def autoscan_users():
    """Пользователи с включённым автосканом: [(chat_id, паттерны, таймфреймы, число монет, настройки)]."""
    result = []
    for key, user in load_users().items():
        if user.get("autoscan") and user.get("patterns"):
            result.append((int(key), user["patterns"], user.get("intervals", DEFAULT_INTERVALS),
                           user.get("coins", DEFAULT_COINS), {**DEFAULT_SETTINGS, **user.get("settings", {})}))
    return result


def get_sent(chat_id):
    """Что уже отправлено автосканом — чтобы не присылать одно и то же дважды."""
    return load_users().get(str(chat_id), {}).get("sent", [])


@locked
def add_sent(chat_id, entries, now):
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    fresh = [e for e in user.get("sent", []) if now - e["at"] < SENT_KEEP_DAYS * 86400]
    user["sent"] = (fresh + entries)[-SENT_MAX:]
    save_users(users)


# --- сколько монет сканировать -------------------------------------------------------------------
DEFAULT_COINS = 50


def get_coins(chat_id):
    return load_users().get(str(chat_id), {}).get("coins", DEFAULT_COINS)


@locked
def set_coins(chat_id, count):
    users = load_users()
    users.setdefault(str(chat_id), {"patterns": []})["coins"] = count
    save_users(users)


# --- настройки поиска ------------------------------------------------------------------------------
DEFAULT_SETTINGS = {"fvg": "significant", "combos_only": False, "similar_only": False, "learn": True,
                    "confluence_only": False, "ob_volume": True}
SETTING_VALUES = {"fvg": ["significant", "all"], "combos_only": [False, True],
                  "similar_only": [False, True], "learn": [True, False],
                  "confluence_only": [False, True], "ob_volume": [True, False]}


def get_settings(chat_id):
    saved = load_users().get(str(chat_id), {}).get("settings", {})
    return {**DEFAULT_SETTINGS, **saved}


@locked
def toggle_setting(chat_id, key):
    """Переключает настройку на следующее значение. Возвращает все настройки."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    settings = {**DEFAULT_SETTINGS, **user.get("settings", {})}
    values = SETTING_VALUES[key]
    settings[key] = values[(values.index(settings[key]) + 1) % len(values)]
    user["settings"] = settings
    save_users(users)
    return settings


# --- новый пользователь -----------------------------------------------------------------------------
@locked
def first_visit(chat_id):
    """True — человек пишет боту впервые (показать приветствие). Отмечает, что приветствие было."""
    users = load_users()
    user = users.get(str(chat_id))
    if user is not None and (user.get("welcomed") or user.get("patterns")):
        return False
    user = users.setdefault(str(chat_id), {"patterns": []})
    user["welcomed"] = True
    save_users(users)
    return True


# --- оценки 👍 / 👎 -------------------------------------------------------------------------------
FEEDBACK_MAX = 2000


def get_feedback(chat_id):
    return load_users().get(str(chat_id), {}).get("feedback", [])


@locked
def add_feedback(chat_id, record):
    """Запоминает оценку. Повторная оценка той же находки заменяет прежнюю."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    feedback = [r for r in user.get("feedback", []) if r["key"] != record["key"]]
    user["feedback"] = (feedback + [record])[-FEEDBACK_MAX:]
    save_users(users)
    return user["feedback"]


@locked
def clear_feedback(chat_id):
    users = load_users()
    users.setdefault(str(chat_id), {"patterns": []})["feedback"] = []
    save_users(users)
