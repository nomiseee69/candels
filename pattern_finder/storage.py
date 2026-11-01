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
