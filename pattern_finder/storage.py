"""Память бота: паттерны, настройки, оценки и подписки пользователей — в базе SQLite (users.db).

Раньше всё лежало в одном файле users.json. Для одного человека это нормально, но когда
пользователей десятки: файл переписывается целиком при каждом клике, и сбой посреди записи
может его испортить. SQLite — это настоящая база данных в одном файле (встроена в Python,
ставить ничего не нужно): запись идёт «транзакцией» — либо целиком, либо никак.

Старый users.json при первом запуске сам переезжает в базу (файл переименовывается в
users.json.migrated — это и есть резервная копия).

Бот работает в несколько потоков (сканирование идёт в фоне), поэтому каждое изменение —
«прочитать → поменять → записать» — делается под замком LOCK: иначе два потока
прочитают старую версию, и изменения одного из них потеряются."""

import functools
import json
import os
import sqlite3
import threading
import time
from contextlib import closing, contextmanager

FILE = "users.json"          # имя старого файла; база лежит рядом: users.db
LOCK = threading.RLock()     # RLock: функция под замком может вызвать другую такую же
BACKUP_DIR = "backups"
BACKUP_KEEP = 7              # сколько последних ежедневных копий базы хранить


def locked(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with LOCK:
            return func(*args, **kwargs)
    return wrapper


def db_path():
    base = FILE[:-5] if FILE.endswith(".json") else FILE
    return base + ".db"


def _connect():
    connection = sqlite3.connect(db_path(), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")       # чтение не ждёт записи
    connection.execute("CREATE TABLE IF NOT EXISTS users (chat_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS payments (id TEXT PRIMARY KEY, chat_id TEXT, source TEXT, "
                       "plan TEXT, amount REAL, currency TEXT, at REAL, status TEXT, data TEXT)")
    return connection


def _migrate():
    """users.json → users.db один раз."""
    if os.path.exists(db_path()) or not os.path.exists(FILE):
        return
    with open(FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    connection = _connect()
    with connection:
        connection.executemany("INSERT OR REPLACE INTO users VALUES (?, ?)",
                               [(key, json.dumps(user, ensure_ascii=False)) for key, user in users.items()])
    connection.close()
    os.replace(FILE, FILE + ".migrated")
    print(f"Перенёс {len(users)} пользователей из {FILE} в базу {db_path()}")


@contextmanager
def transaction():
    """Запись в SQLite с тем же замком, что и у остальных изменений пользователей."""
    with LOCK:
        _migrate()
        with closing(_connect()) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            yield connection


def update_user_in(connection, chat_id, change):
    row = connection.execute("SELECT data FROM users WHERE chat_id = ?", (str(chat_id),)).fetchone()
    user = json.loads(row[0]) if row else {"patterns": []}
    result = change(user)
    connection.execute("INSERT OR REPLACE INTO users VALUES (?, ?)",
                       (str(chat_id), json.dumps(user, ensure_ascii=False)))
    return result


@locked
def load_users():
    _migrate()
    if not os.path.exists(db_path()):
        return {}
    connection = _connect()
    rows = connection.execute("SELECT chat_id, data FROM users").fetchall()
    connection.close()
    users = {key: json.loads(data) for key, data in rows}
    # старые записи без id: используем путь к фото как id
    for user in users.values():
        for p in user.get("patterns", []):
            p.setdefault("id", p["photo"])
            p.setdefault("note", "")
    return users


@locked
def save_users(users):
    """Записывает всех пользователей одной транзакцией: сбой посреди записи базу не испортит."""
    connection = _connect()
    with connection:
        old = {row[0] for row in connection.execute("SELECT chat_id FROM users")}
        connection.executemany("INSERT OR REPLACE INTO users VALUES (?, ?)",
                               [(str(key), json.dumps(user, ensure_ascii=False)) for key, user in users.items()])
        gone = old - {str(k) for k in users}
        connection.executemany("DELETE FROM users WHERE chat_id = ?", [(k,) for k in gone])
    connection.close()


def get_user(chat_id):
    return load_users().get(str(chat_id), {})


@locked
def update_user(chat_id, change):
    """change(user) меняет запись пользователя на месте; результат change возвращается."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    result = change(user)
    save_users(users)
    return result


# --- платежи: журнал для статистики и проверки повторов ------------------------------------------
@locked
def add_payment(payment_id, chat_id, source, plan, amount, currency, status="succeeded", data=None):
    """True — платёж новый и записан; False — такой уже был (повторное уведомление)."""
    connection = _connect()
    with connection:
        inserted = add_payment_in(connection, payment_id, chat_id, source, plan, amount, currency,
                                  status=status, data=data)
    connection.close()
    return inserted


def add_payment_in(connection, payment_id, chat_id, source, plan, amount, currency, *,
                   status="succeeded", data=None):
    cursor = connection.execute(
        "INSERT OR IGNORE INTO payments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(payment_id), str(chat_id), source, plan, float(amount), currency, time.time(), status,
         json.dumps(data or {}, ensure_ascii=False)))
    return cursor.rowcount == 1


@locked
def set_payment_status(payment_id, status):
    """Пометить платёж (например, "refunded"). True — платёж был и статус поменялся."""
    if not os.path.exists(db_path()):
        return False
    connection = _connect()
    with connection:
        cursor = connection.execute("UPDATE payments SET status = ? WHERE id = ? AND status != ?",
                                    (status, str(payment_id), status))
    connection.close()
    return cursor.rowcount == 1


def payments(since=0):
    if not os.path.exists(db_path()):
        return []
    connection = _connect()
    rows = connection.execute("SELECT id, chat_id, source, plan, amount, currency, at, status FROM payments "
                              "WHERE at >= ? ORDER BY at", (since,)).fetchall()
    connection.close()
    keys = ("id", "chat_id", "source", "plan", "amount", "currency", "at", "status")
    return [dict(zip(keys, row)) for row in rows]


# --- резервные копии -------------------------------------------------------------------------------
@locked
def backup(now=None):
    """Копия базы в backups/users-ГГГГ-ММ-ДД.db (безопасно даже во время работы бота).
    Хранит BACKUP_KEEP последних. Возвращает путь копии или None (базы ещё нет)."""
    _migrate()
    if not os.path.exists(db_path()):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d", time.localtime(now or time.time()))
    target = os.path.join(BACKUP_DIR, f"users-{stamp}.db")
    source = _connect()
    copy = sqlite3.connect(target)
    with copy:
        source.backup(copy)
    copy.close()
    source.close()
    old = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("users-") and f.endswith(".db"))
    for name in old[:-BACKUP_KEEP]:
        os.remove(os.path.join(BACKUP_DIR, name))
    return target


@locked
def add_pattern(chat_id, name, photo_path, note="", kind=None, reference=None, shape=None, builtin=False,
                fingerprint=None, direction=None, shape_window=None, neck_template=None):
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
    for key, value in (("direction", direction), ("shape_window", shape_window), ("neck_template", neck_template)):
        if value is not None:
            entry[key] = value
    user["patterns"].append(entry)
    save_users(users)
    return pattern_id


def get_patterns(chat_id):
    return load_users().get(str(chat_id), {}).get("patterns", [])


@locked
def set_spec(chat_id, pattern_id, spec):
    """Описание паттерна словами (language.parse) — к примеру pattern_id. Возвращает пример или None."""
    users = load_users()
    for p in users.get(str(chat_id), {}).get("patterns", []):
        if p["id"] == pattern_id:
            if spec:
                p["spec"] = spec
            else:
                p.pop("spec", None)
            save_users(users)
            return p
    return None


def update_pattern(chat_id, pattern_id, **fields):
    """Поменять поля примера (например, направление, если бот понял его наоборот)."""
    users = load_users()
    for p in users.get(str(chat_id), {}).get("patterns", []):
        if p["id"] == pattern_id:
            p.update(fields)
            save_users(users)
            return p
    return None


def get_pattern(chat_id, pattern_id):
    return next((p for p in get_patterns(chat_id) if p["id"] == pattern_id), None)


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
                    "confluence_only": False, "ob_volume": True, "quality": "normal"}
SETTING_VALUES = {"fvg": ["significant", "all"], "combos_only": [False, True],
                  "similar_only": [False, True], "learn": [True, False],
                  "confluence_only": [False, True], "ob_volume": [True, False],
                  "quality": ["normal", "strict", "off"]}


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
