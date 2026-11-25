"""Shared Stars checkout guard, independent of how an invoice was opened."""
import json
import time

import storage

HOLD = 5 * 60
BLOCKED = "Доступ Pro уже оплачен. Новый тариф можно выбрать после окончания доступа."


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS pro_checkout_gate (
        buyer TEXT PRIMARY KEY, query TEXT NOT NULL, until REAL NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS pro_conflicts (
        charge TEXT PRIMARY KEY, buyer TEXT NOT NULL, recurring INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending', attempted REAL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS pro_renewal_cancelled (
        charge TEXT PRIMARY KEY)""")


def paid_until(db, buyer, now):
    row = db.execute("SELECT data FROM users WHERE chat_id=?", (str(buyer),)).fetchone()
    sub = (json.loads(row[0]).get("sub") or {}) if row else {}
    until = sub.get("until") or 0
    return until if until > now and sub.get("source") in ("stars", "yookassa") else 0


def reserve(db, event, now):
    buyer = (event.get("from") or {}).get("id")
    query = event.get("id")
    if type(buyer) is not int or buyer <= 0 or not query:
        return False
    schema(db)
    if paid_until(db, buyer, now):
        return False
    existing = db.execute("SELECT query,until FROM pro_checkout_gate WHERE buyer=?", (str(buyer),)).fetchone()
    if existing and existing[1] > now:
        return existing[0] == query
    db.execute("INSERT OR REPLACE INTO pro_checkout_gate VALUES (?,?,?)", (str(buyer), query, now + HOLD))
    return True


def release(db, buyer):
    schema(db)
    db.execute("DELETE FROM pro_checkout_gate WHERE buyer=?", (str(buyer),))


def renewal(event):
    return bool(event.get("is_recurring") and not event.get("is_first_recurring"))


def track_conflict(db, buyer, payment, *, review=False):
    schema(db)
    db.execute("INSERT OR IGNORE INTO pro_conflicts(charge,buyer,recurring,state) VALUES (?,?,?,?)",
               (payment["telegram_payment_charge_id"], str(buyer),
                int(bool(payment.get("subscription_expiration_date"))), "review" if review else "pending"))


def refund_extra(db, buyer, charge):
    schema(db)
    return bool(db.execute("UPDATE pro_conflicts SET state='refunded' WHERE charge=? AND buyer=?",
                           (charge, str(buyer))).rowcount)


def process_conflicts(api, now=None):
    now = time.time() if now is None else now
    with storage.transaction() as db:
        schema(db)
        rows = db.execute("SELECT charge,buyer,recurring FROM pro_conflicts WHERE state='pending' "
                          "AND (attempted IS NULL OR attempted<?) LIMIT 10", (now-HOLD,)).fetchall()
        db.executemany("UPDATE pro_conflicts SET attempted=? WHERE charge=?", [(now, row[0]) for row in rows])
    for charge, buyer, recurring in rows:
        args = {"user_id": int(buyer), "telegram_payment_charge_id": charge}
        try:
            if recurring and not api("editUserStarSubscription", {**args, "is_canceled": True}).get("ok"):
                mark_review(charge)
                continue
            refunded = api("refundStarPayment", args).get("ok")
            if not refunded:
                mark_review(charge)
                continue
            with storage.transaction() as db:
                refund_extra(db, buyer, charge)
                db.execute("UPDATE payments SET status='refunded' WHERE id=? AND status='conflict'", (charge,))
        except Exception:
            # Retry after the lease. A refund update can also reconcile the operation.
            continue


def mark_review(charge):
    with storage.transaction() as db:
        db.execute("UPDATE pro_conflicts SET state='review' WHERE charge=? AND state='pending'", (charge,))


def renewal_conflicts(db, buyer, payload, now):
    if not paid_until(db, buyer, now):
        return False
    latest = db.execute("SELECT plan,data FROM payments WHERE chat_id=? AND status='succeeded' "
                        "ORDER BY at DESC LIMIT 1", (str(buyer),)).fetchone()
    if not latest:
        return False
    previous = json.loads(latest[1]).get("invoice_payload")
    return latest[0] != "month" or bool(previous and previous != payload)


def recurring_charge(db, buyer):
    """Only a confirmed initial Stars payment identifies a subscription we can manage."""
    latest = db.execute("SELECT data FROM payments WHERE chat_id=? AND source='stars' AND plan='month' "
                        "AND status='succeeded' ORDER BY at DESC LIMIT 1", (str(buyer),)).fetchone()
    if not latest:
        return None
    payload = json.loads(latest[0]).get("invoice_payload")
    rows = db.execute("SELECT id,data FROM payments WHERE chat_id=? AND source='stars' AND plan='month' "
                      "AND status='succeeded' ORDER BY at DESC", (str(buyer),))
    for charge, data in rows:
        event = json.loads(data)
        if (event.get("invoice_payload") == payload and event.get("is_first_recurring")
                and event.get("subscription_expiration_date")):
            return charge
    return None
