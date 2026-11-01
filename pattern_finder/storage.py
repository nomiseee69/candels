"""Память бота: паттерны пользователей в users.json."""

import json
import os
import time

FILE = "users.json"


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


def save_users(users):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def add_pattern(chat_id, name, photo_path, note="", kind=None):
    """Сохраняет паттерн и возвращает его id.
    kind — тип фигуры с рисунка пользователя (например, "symmetric"), если линии нарисованы."""
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    pattern_id = str(int(time.time() * 1000))
    user["patterns"].append({
        "id": pattern_id, "name": name, "photo": photo_path, "note": note, "kind": kind,
    })
    save_users(users)
    return pattern_id


def get_patterns(chat_id):
    return load_users().get(str(chat_id), {}).get("patterns", [])


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


def toggle_autoscan(chat_id):
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    user["autoscan"] = not user.get("autoscan", False)
    save_users(users)
    return user["autoscan"]


def autoscan_users():
    """Пользователи с включённым автосканом: [(chat_id, паттерны, таймфреймы)]."""
    result = []
    for key, user in load_users().items():
        if user.get("autoscan") and user.get("patterns"):
            result.append((int(key), user["patterns"], user.get("intervals", DEFAULT_INTERVALS)))
    return result


def get_sent(chat_id):
    """Что уже отправлено автосканом — чтобы не присылать одно и то же дважды."""
    return load_users().get(str(chat_id), {}).get("sent", [])


def add_sent(chat_id, entries, now):
    users = load_users()
    user = users.setdefault(str(chat_id), {"patterns": []})
    fresh = [e for e in user.get("sent", []) if now - e["at"] < SENT_KEEP_DAYS * 86400]
    user["sent"] = (fresh + entries)[-SENT_MAX:]
    save_users(users)
