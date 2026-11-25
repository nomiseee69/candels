"""User-bound Stars orders issued by the Mini App."""
import os
import secrets
import time
from urllib.parse import urlsplit

import storage

PREFIX = "ma_pro:"
TTL = 900


def app_url():
    url = os.getenv("MINI_APP_URL", "")
    parsed = urlsplit(url)
    return url if parsed.scheme == "https" and parsed.hostname and not parsed.username else ""


def is_order(event):
    return str(event.get("invoice_payload", "")).startswith(PREFIX)


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS mini_orders (
        payload TEXT PRIMARY KEY, buyer INTEGER NOT NULL, plan TEXT NOT NULL,
        amount INTEGER NOT NULL CHECK(amount>0), expires REAL NOT NULL,
        checkout TEXT UNIQUE, initial_charge TEXT UNIQUE)""")


def create(buyer, plan, amount, now=None):
    now = time.time() if now is None else now
    payload = PREFIX + secrets.token_urlsafe(24)
    with storage.transaction() as db:
        schema(db)
        count = db.execute("SELECT COUNT(*) FROM mini_orders WHERE buyer=? AND expires>?",
                           (buyer, now)).fetchone()[0]
        if count >= 6:
            raise ValueError("Слишком много открытых счетов. Повтори позже.")
        db.execute("DELETE FROM mini_orders WHERE expires<=? AND checkout IS NULL", (now,))
        db.execute("INSERT INTO mini_orders VALUES (?,?,?,?,?,NULL,NULL)", (payload, buyer, plan, amount, now + TTL))
    return payload


def order(db, buyer, event):
    schema(db)
    return db.execute("SELECT plan,expires,checkout,initial_charge FROM mini_orders "
                      "WHERE payload=? AND buyer=? AND amount=? AND ?='XTR'",
                      (event.get("invoice_payload"), buyer, event.get("total_amount"), event.get("currency"))).fetchone()


def checkout(event, now=None):
    now = time.time() if now is None else now
    with storage.transaction() as db:
        found = order(db, (event.get("from") or {}).get("id"), event)
        if not found or found[1] <= now or found[3] or (found[2] and found[2] != event["id"]):
            return False
        db.execute("UPDATE mini_orders SET checkout=? WHERE payload=?", (event["id"], event["invoice_payload"]))
        return True


def settle(db, buyer, payment, charge):
    found = order(db, buyer, payment)
    if not found or not found[2]:
        return None
    plan, _, _, first = found
    renewal = payment.get("is_recurring") and not payment.get("is_first_recurring")
    if renewal:
        return plan if first and plan == "month" and payment.get("subscription_expiration_date") else None
    if first or (plan == "month" and not payment.get("subscription_expiration_date")):
        return None
    db.execute("UPDATE mini_orders SET initial_charge=? WHERE payload=?", (charge, payment["invoice_payload"]))
    return plan
