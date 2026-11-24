"""Одноразовые персональные скидки на Pro через Telegram Stars."""

import json
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

import billing
import storage

PREFIX = "pro_discount:"
INVOICE_TTL = billing.DAY
CHECKOUT_TTL = 5 * 60


@contextmanager
def transaction():
    with storage.transaction() as db:
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS pro_discounts (
            id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, percent INTEGER NOT NULL CHECK(percent BETWEEN 1 AND 99),
            issued_by INTEGER NOT NULL, issued_at REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('active', 'used', 'revoked')),
            closed_at REAL, charge TEXT UNIQUE
        )""")
        db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_active_pro_discount
                      ON pro_discounts(user_id) WHERE status = 'active'""")
        db.execute("""CREATE TABLE IF NOT EXISTS pro_discount_orders (
            token TEXT PRIMARY KEY, discount_id INTEGER NOT NULL REFERENCES pro_discounts(id),
            plan TEXT NOT NULL, amount INTEGER NOT NULL CHECK(amount > 0), expires_at REAL NOT NULL,
            query_id TEXT UNIQUE, reserved_until REAL, charge TEXT UNIQUE
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS pro_discount_order_reservation
                      ON pro_discount_orders(discount_id, reserved_until)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pro_discount_refunds (
            charge TEXT PRIMARY KEY, user_id INTEGER NOT NULL, refunded_at REAL, attempted_at REAL
        )""")
        yield db


def resolve_user(value, api_post):
    """Искать только в базе. Уточнить username через getChat по уже известному ID."""
    value = value.strip()
    users = storage.load_users()
    if re.fullmatch(r"[0-9]{1,16}", value):
        key = str(int(value))
        return int(key) if int(key) > 0 and key in users else None
    name = value.removeprefix("@").lower()
    if not re.fullmatch(r"[a-z0-9_]{1,32}", name):
        return None
    matches = [int(key) for key, user in users.items() if key.isdecimal() and int(key) > 0
               and (user.get("telegram_username", user.get("username")) or "").lower() == name]
    if len(matches) != 1:
        return None
    user_id = matches[0]
    answer = api_post("getChat", {"chat_id": user_id})
    chat = answer.get("result") or {}
    if (not answer.get("ok") or chat.get("id") != user_id or chat.get("type") != "private"
            or (chat.get("username") or "").lower() != name):
        return None
    if users[str(user_id)].get("telegram_username") != chat["username"]:
        storage.update_user(user_id, lambda user: user.update(telegram_username=chat["username"]))
    return user_id


def price(base, percent):
    return (base * (100 - percent) + 99) // 100


def _active(db, user_id):
    return db.execute("SELECT * FROM pro_discounts WHERE user_id = ? AND status = 'active'",
                      (user_id,)).fetchone()


def active(user_id):
    with transaction() as db:
        return _active(db, user_id)


def list_active(offset=0, limit=9):
    with transaction() as db:
        return db.execute("SELECT * FROM pro_discounts WHERE status = 'active' ORDER BY id DESC LIMIT ? OFFSET ?",
                          (limit, offset)).fetchall()


def _reserved(db, discount_id, now):
    return db.execute("SELECT token FROM pro_discount_orders WHERE discount_id = ? AND reserved_until > ?",
                      (discount_id, now)).fetchone()


def issue(admin_id, user_id, percent, expected_id, now=None):
    if not billing.is_admin(admin_id) or type(percent) is not int or not 1 <= percent <= 99:
        return "invalid"
    now = time.time() if now is None else now
    with transaction() as db:
        if user_id <= 0 or not db.execute("SELECT 1 FROM users WHERE chat_id = ?", (str(user_id),)).fetchone():
            return "unknown"
        old = _active(db, user_id)
        if (old["id"] if old else None) != expected_id:
            return "changed"
        if old:
            if _reserved(db, old["id"], now):
                return "busy"
            db.execute("UPDATE pro_discounts SET status = 'revoked', closed_at = ? WHERE id = ?", (now, old["id"]))
        db.execute("INSERT INTO pro_discounts (user_id, percent, issued_by, issued_at, status) "
                   "VALUES (?, ?, ?, ?, 'active')", (user_id, percent, admin_id, now))
        return "issued"


def revoke(admin_id, user_id, expected_id, now=None):
    if not billing.is_admin(admin_id):
        return "invalid"
    now = time.time() if now is None else now
    with transaction() as db:
        old = _active(db, user_id)
        if not old or old["id"] != expected_id:
            return "changed"
        if _reserved(db, old["id"], now):
            return "busy"
        db.execute("UPDATE pro_discounts SET status = 'revoked', closed_at = ? WHERE id = ?", (now, expected_id))
        return "revoked"


def is_discount(payment):
    return (payment.get("invoice_payload") or "").startswith(PREFIX)


def invoice(user_id, plan, now=None):
    """Новый одноразовый счёт. Ни цена, ни скидка не приходят из кнопки."""
    if plan not in billing.PLANS:
        return None
    now = time.time() if now is None else now
    with transaction() as db:
        discount = _active(db, user_id)
        if not discount:
            return None
        _, base, _, title = billing.PLANS[plan]
        amount = price(base, discount["percent"])
        token = PREFIX + secrets.token_urlsafe(24)
        db.execute("INSERT INTO pro_discount_orders (token, discount_id, plan, amount, expires_at) "
                   "VALUES (?, ?, ?, ?, ?)", (token, discount["id"], plan, amount, now + INVOICE_TTL))
        return {"chat_id": user_id, "title": title, "payload": token, "currency": "XTR", "provider_token": "",
                "description": f"Одна оплата Pro со скидкой {discount['percent']}%.",
                "start_parameter": "pro", "prices": json.dumps([{"label": title, "amount": amount}])}


def _order(db, payment):
    return db.execute("""SELECT o.*, d.user_id, d.status, d.percent FROM pro_discount_orders o
                       JOIN pro_discounts d ON d.id = o.discount_id WHERE o.token = ?""",
                      (payment["invoice_payload"],)).fetchone()


def _matches(order, user_id, payment):
    return (order and order["user_id"] == user_id and payment.get("currency") == "XTR"
            and type(payment.get("total_amount")) is int and payment["total_amount"] == order["amount"])


def pre_checkout(query, now=None):
    now = time.time() if now is None else now
    user_id = (query.get("from") or {}).get("id")
    with transaction() as db:
        order = _order(db, query)
        if (not _matches(order, user_id, query) or order["status"] != "active"
                or order["expires_at"] <= now or order["plan"] not in billing.PLANS
                or order["amount"] != price(billing.PLANS[order["plan"]][1], order["percent"])):
            return "stale"
        if order["query_id"] == query["id"]:
            return "ok" if order["reserved_until"] > now else "stale"
        if _reserved(db, order["discount_id"], now):
            return "busy"
        if order["query_id"]:
            return "stale"
        db.execute("UPDATE pro_discount_orders SET query_id = ?, reserved_until = ? WHERE token = ?",
                   (query["id"], now + CHECKOUT_TTL, order["token"]))
        return "ok"


def settle(user_id, payment, now=None):
    """Записать оплату, погасить скидку и продлить Pro одной транзакцией."""
    charge = payment.get("telegram_payment_charge_id")
    if not charge or payment.get("currency") != "XTR":
        return "invalid", None
    now = time.time() if now is None else now
    with transaction() as db:
        if (db.execute("SELECT 1 FROM payments WHERE id = ?", (charge,)).fetchone()
                or db.execute("SELECT 1 FROM pro_discount_refunds WHERE charge = ?", (charge,)).fetchone()):
            return "duplicate", None
        order = _order(db, payment)
        if (not _matches(order, user_id, payment) or not order["query_id"]
                or order["status"] != "active" or order["plan"] not in billing.PLANS
                or payment.get("is_recurring") or payment.get("subscription_expiration_date")):
            db.execute("INSERT INTO pro_discount_refunds (charge, user_id) VALUES (?, ?)", (charge, user_id))
            return "refund", None

        plan = order["plan"]
        storage.add_payment_in(db, charge, user_id, "stars", plan, order["amount"], "XTR", data=payment)
        until = storage.update_user_in(db, user_id, lambda user: billing.extend_user(
            user, billing.PLANS[plan][0], "stars", None, (user.get("sub") or {}).get("recurring", False), now))
        db.execute("UPDATE pro_discount_orders SET charge = ?, reserved_until = NULL WHERE token = ?",
                   (charge, order["token"]))
        db.execute("UPDATE pro_discounts SET status = 'used', closed_at = ?, charge = ? WHERE id = ?",
                   (now, charge, order["discount_id"]))
        return "paid", until


def mark_extra_refunded(user_id, charge):
    """Отличить возврат лишнего платежа от возврата оплаченного Pro."""
    with transaction() as db:
        result = db.execute("UPDATE pro_discount_refunds SET refunded_at = COALESCE(refunded_at, ?) "
                            "WHERE charge = ? AND user_id = ?", (time.time(), charge, user_id))
        return result.rowcount > 0


def retry_refunds(api_post, now=None):
    """При задержке Telegram после истечения резерва может подтвердить лишнюю оплату."""
    with transaction() as db:
        now = time.time() if now is None else now
        pending = db.execute("SELECT charge, user_id FROM pro_discount_refunds WHERE refunded_at IS NULL "
                             "AND (attempted_at IS NULL OR attempted_at <= ?) ORDER BY attempted_at LIMIT 5",
                             (now - 60,)).fetchall()
        db.executemany("UPDATE pro_discount_refunds SET attempted_at = ? WHERE charge = ?",
                       [(now, row["charge"]) for row in pending])
    for row in pending:
        reply = api_post("refundStarPayment", {"user_id": row["user_id"], "telegram_payment_charge_id": row["charge"]})
        if reply.get("ok") or "CHARGE_ALREADY_REFUNDED" in reply.get("description", ""):
            mark_extra_refunded(row["user_id"], row["charge"])
