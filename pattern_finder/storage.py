import json
import os

FILE = "users.json"


def load_users():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def add_pattern(chat_id, name, photo_path):
    users = load_users()
    key = str(chat_id)
    if key not in users:
        users[key] = {"patterns": []}
    users[key]["patterns"].append({"name": name, "photo": photo_path})
    save_users(users)
def delete_pattern(chat_id, index):
    users = load_users()
    key = str(chat_id)
    if key in users:
        users[key]["patterns"].pop(index)
        save_users(users)

def get_patterns(chat_id):
    users = load_users()
    return users.get(str(chat_id), {}).get("patterns", [])