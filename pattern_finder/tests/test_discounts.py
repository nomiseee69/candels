import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

import billing
import bot
import discounts
import storage

ADMIN, USER, OTHER = 1, 10001, 10002
NOW = 1_800_000_000


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setenv("ADMIN_IDS", str(ADMIN))
    monkeypatch.setattr(bot, "discount_drafts", {})
    monkeypatch.setattr(bot, "alert_admins", lambda *args: None)
    monkeypatch.delenv("YOOKASSA_SHOP_ID", raising=False)
    for key, name in ((ADMIN, "admin"), (USER, "known"), (OTHER, "other")):
        storage.update_user(key, lambda u: u.update(telegram_username=name, sub={"trial_used": True}))


def issue(percent=50):
    assert discounts.issue(ADMIN, USER, percent, None, NOW) == "issued"
    return discounts.active(USER)["id"]


def query(plan="month", query_id="q", now=NOW):
    invoice = discounts.invoice(USER, plan, now)
    return invoice, {"id": query_id, "from": {"id": USER}, "invoice_payload": invoice["payload"],
                     "currency": "XTR", "total_amount": json.loads(invoice["prices"])[0]["amount"]}


def paid_event(checkout, charge="paid"):
    return {key: checkout[key] for key in ("invoice_payload", "currency", "total_amount")} | {
        "telegram_payment_charge_id": charge}


def send_text(user_id, text, username=None):
    sender = {"id": user_id}
    if username:
        sender["username"] = username
    bot.handle_message({"chat": {"id": user_id, "type": "private"}, "from": sender, "text": text})


@pytest.fixture
def telegram(monkeypatch):
    replies, calls, decisions = [], [], []
    monkeypatch.setattr(bot, "answer_callback", lambda *args: None)
    monkeypatch.setattr(bot, "answer_pre_checkout", lambda *args: decisions.append(args))
    monkeypatch.setattr(bot, "show_view", lambda user, mid, view, media=False: replies.append((user, *view)))
    monkeypatch.setattr(bot, "send_message", lambda text, user, keyboard=None: replies.append((user, text, keyboard)))
    monkeypatch.setattr(bot, "background", lambda func, *args: func(*args))
    def api(method, data):
        calls.append((method, data))
        chat_id = data.get("chat_id")
        if method == "getChat":
            return {"ok": True, "result": {"id": chat_id, "type": "private",
                                           "username": storage.get_user(chat_id).get("telegram_username")}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bot, "api_post", api)
    monkeypatch.setattr(billing, "stars_links", lambda api: {"month": "https://t.me/regular"})
    return replies, calls, decisions


def press(data, user=ADMIN, sender=None):
    bot.handle_callback({"id": "cb", "from": {"id": user if sender is None else sender}, "data": data,
                         "message": {"chat": {"id": user, "type": "private"}, "message_id": 3}})


def button(keyboard, label):
    return next(b["callback_data"] for row in keyboard["inline_keyboard"] for b in row if label in b["text"])


def test_admin_issue_replace_and_revoke_from_buttons(telegram):
    replies, calls, _ = telegram
    send_text(ADMIN, "/admin")
    press(button(replies[-1][2], "Скидки"))
    press(button(replies[-1][2], "Выдать"))
    send_text(ADMIN, "@unknown")
    assert "/start" in replies[-1][1]
    send_text(ADMIN, "@KNOWN")
    send_text(ADMIN, "100")
    assert "от 1 до 99" in replies[-1][1]
    send_text(ADMIN, "50%")
    confirmation = button(replies[-1][2], "Подтвердить")
    assert "@known" in replies[-1][1] and discounts.active(USER) is None
    press(confirmation, user=OTHER)
    press(confirmation, sender=OTHER)
    assert discounts.active(USER) is None
    press(confirmation)
    assert discounts.active(USER)["percent"] == 50
    assert replies[-1][0] == USER and "50%" in replies[-1][1]
    press("adm:d:list:0")
    assert "@known" in replies[-1][1] and "активна" in replies[-1][1]

    press("adm:d:new")
    send_text(ADMIN, str(USER))
    send_text(ADMIN, "25")
    assert "заменит" in replies[-1][1] and discounts.active(USER)["percent"] == 50
    press(button(replies[-1][2], "Отмена"))
    assert discounts.active(USER)["percent"] == 50
    press("adm:d:new")
    send_text(ADMIN, str(USER))
    send_text(ADMIN, "25")
    press(button(replies[-1][2], "Подтвердить"))
    assert discounts.active(USER)["percent"] == 25

    press("adm:d:pick:0")
    press(button(replies[-1][2], "@known"))
    press(button(replies[-1][2], "Подтвердить"))
    assert discounts.active(USER) is None
    with discounts.transaction() as db:
        assert [r["status"] for r in db.execute("SELECT status FROM pro_discounts ORDER BY id")] == ["revoked", "revoked"]
    assert calls == [("getMyStarBalance", {}), ("getChat", {"chat_id": USER})]


def test_resolve_numeric_id_and_only_observed_usernames(telegram):
    assert discounts.resolve_user(str(USER), bot.api_post) == USER
    assert discounts.resolve_user("999999", bot.api_post) is None
    assert discounts.resolve_user("-10001", bot.api_post) is None
    assert not telegram[1]
    assert discounts.issue(OTHER, USER, 50, None, NOW) == "invalid"
    assert discounts.issue(ADMIN, 999999, 50, None, NOW) == "unknown"
    issue()
    storage.update_user(99999, lambda u: u.update(username="legacy"))
    assert discounts.resolve_user("@legacy", bot.api_post) is None
    assert discounts.resolve_user("@legacy", lambda method, data: {
        "ok": True, "result": {"id": 99999, "type": "private", "username": "Legacy"}}) == 99999
    bot.remember_username({"chat": {"id": USER}, "from": {"id": USER, "first_name": "known"}})
    assert discounts.resolve_user("@known", bot.api_post) is None
    bot.remember_username({"chat": {"id": OTHER}, "from": {"id": OTHER, "username": "known"}})
    assert discounts.resolve_user("@known", bot.api_post) == OTHER and discounts.active(OTHER) is None
    assert discounts.active(USER)["percent"] == 50


@pytest.mark.parametrize(("percent", "amount"), [(10, 181), (25, 151), (50, 101), (99, 3)])
def test_price_rounds_up_and_is_used_in_actual_invoice(percent, amount, monkeypatch, telegram):
    plan = (30, 201, 990, "Pro на 30 дней")
    monkeypatch.setitem(billing.PLANS, "month", plan)
    issue(percent)
    replies, calls, _ = telegram
    press("sub", user=USER)
    assert "201 ⭐" in replies[-1][1] and f"твоя цена {amount} ⭐" in replies[-1][1]
    press("disc_buy:month", user=USER, sender=OTHER)
    press("disc_buy:month:1", user=USER)
    assert calls == []
    press(button(replies[0][2], "Pro на 30 дней"), user=USER)
    [(method, data)] = calls
    assert method == "sendInvoice" and data["chat_id"] == USER and data["currency"] == "XTR"
    assert json.loads(data["prices"]) == [{"label": plan[3], "amount": amount}]
    assert "subscription_period" not in data and len(data["payload"].encode()) <= 128
    text, keyboard = bot.subscription_view(OTHER)
    assert "персональная скидка" not in text
    assert keyboard["inline_keyboard"][0][0]["url"] == "https://t.me/regular"


def test_invoice_send_failure_and_checkout_abandonment_do_not_spend_discount(monkeypatch, telegram):
    issue()
    monkeypatch.setattr(bot, "api_post", lambda *args: {"ok": False})
    bot.send_discount_invoice(USER, "year")
    assert "Скидка сохранена" in telegram[0][-1][1]
    _, first = query(query_id="first")
    _, second = query(query_id="second")
    assert discounts.pre_checkout(first, NOW) == "ok"
    assert discounts.pre_checkout(second, NOW) == "busy"
    assert discounts.active(USER)["percent"] == 50 and not billing.history(USER)
    assert discounts.pre_checkout(second, NOW + discounts.CHECKOUT_TTL) == "ok"


def test_paid_consumes_once_with_legacy_recurring_subscription(monkeypatch, telegram):
    discount_id = issue()
    storage.update_user(USER, lambda u: u["sub"].update(until=NOW + billing.DAY, recurring=True))
    _, first = query("year")
    monkeypatch.setattr(discounts.time, "time", lambda: NOW)
    bot.handle_pre_checkout(first)
    assert telegram[2][-1] == ("q", True, None)
    payment = paid_event(first)
    message = {"chat": {"id": USER, "type": "private"}, "from": {"id": USER}, "successful_payment": payment}
    bot.handle_payment(message)
    bot.handle_payment(message)
    state = billing.status(USER, NOW)
    assert state["until"] == NOW + (1 + billing.PLANS["year"][0]) * billing.DAY and state["recurring"]
    assert len(billing.history(USER)) == 1
    assert discounts.active(USER) is None and "персональная скидка" not in bot.subscription_view(USER)[0]
    assert discounts.invoice(USER, "month", NOW) is None and discounts.pre_checkout(first, NOW) == "stale"
    assert len([msg for _, msg, _ in telegram[0] if "Скидка использована" in msg]) == 1
    with discounts.transaction() as db:
        assert tuple(db.execute("SELECT status, charge FROM pro_discounts WHERE id = ?", (discount_id,)).fetchone()) == (
            "used", "paid")


@pytest.mark.parametrize("changes", [{"currency": "USD"}, {"total_amount": 1}, {"from": {"id": OTHER}}])
def test_pre_checkout_rejects_invalid_amount_currency_and_other_user(changes):
    issue()
    _, checkout = query()
    assert discounts.pre_checkout(checkout | changes, NOW) == "stale"
    assert discounts.pre_checkout(checkout, NOW) == "ok"


def test_settlement_rechecks_paid_amount_and_identity():
    issue()
    _, checkout = query()
    payment = paid_event(checkout)
    assert discounts.pre_checkout(checkout, NOW) == "ok"
    assert discounts.settle(OTHER, payment, NOW) == ("refund", None)
    assert discounts.settle(USER, paid_event(checkout, "bad_sum") | {"total_amount": 1}, NOW) == ("refund", None)
    assert discounts.settle(USER, paid_event(checkout, "bad_currency") | {"currency": "USD"}, NOW) == ("invalid", None)
    assert discounts.active(USER) and not billing.history(USER)
    assert not billing.history(OTHER)


def test_two_concurrent_checkouts_and_late_double_payment_refund(telegram):
    issue()
    _, first = query("month", "first")
    _, second = query("year", "second")
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda q: discounts.pre_checkout(q, NOW), (first, second)))
    assert sorted(results) == ["busy", "ok"]
    early, late = (first, second) if results[0] == "ok" else (second, first)
    finish = NOW + discounts.CHECKOUT_TTL
    assert discounts.pre_checkout(early, NOW) == "ok"  # same query redelivered
    assert discounts.pre_checkout(late, finish) == "ok"  # earlier Telegram response delayed
    assert discounts.settle(USER, paid_event(late, "valid"), finish)[0] == "paid"
    assert discounts.settle(USER, paid_event(early, "extra"), finish) == ("refund", None)
    assert discounts.settle(USER, paid_event(early, "extra"), finish) == ("duplicate", None)
    state = storage.get_user(USER)["sub"]
    with discounts.transaction() as db:
        assert db.execute("SELECT charge FROM pro_discount_refunds").fetchone()[0] == "extra"
    clock = [finish]
    attempts = []

    def api(method, data):
        attempts.append((method, data))
        return {"ok": len(attempts) == 2}

    discounts.retry_refunds(api, now=clock[0])
    discounts.retry_refunds(api, now=clock[0] + 10)
    discounts.retry_refunds(api, now=clock[0] + 60)
    discounts.retry_refunds(api, now=clock[0] + 120)
    assert attempts == [("refundStarPayment", {"user_id": USER, "telegram_payment_charge_id": "extra"})] * 2
    refund = paid_event(early, "extra")
    bot.handle_refunded({"chat": {"id": USER}, "refunded_payment": refund})
    assert storage.get_user(USER)["sub"] == state
    assert [p["id"] for p in billing.history(USER)] == ["valid"]


def test_stale_invoices_after_revocation_replacement_expiry_and_price_change(monkeypatch):
    original = issue()
    _, first = query(query_id="first")
    assert discounts.revoke(ADMIN, USER, original, NOW) == "revoked"
    assert discounts.pre_checkout(first, NOW) == "stale"
    current = issue(10)
    _, second = query(query_id="second")
    assert discounts.issue(ADMIN, USER, 25, original, NOW) == "changed"
    assert discounts.issue(ADMIN, USER, 25, current, NOW) == "issued"
    assert discounts.pre_checkout(second, NOW) == "stale"
    _, third = query(query_id="third")
    assert discounts.pre_checkout(third, NOW + discounts.INVOICE_TTL) == "stale"
    monkeypatch.setitem(billing.PLANS, "month", (30, 800, 990, "Pro"))
    assert discounts.pre_checkout(third, NOW) == "stale"


def test_atomic_rollback_and_persistence_after_restart(monkeypatch):
    discount_id = issue()
    _, checkout = query()
    assert discounts.pre_checkout(checkout, NOW) == "ok"
    assert discounts.revoke(ADMIN, USER, discount_id, NOW) == "busy"
    assert discounts.issue(ADMIN, USER, 15, discount_id, NOW) == "busy"
    with monkeypatch.context() as failure:
        def fail(*args):
            raise sqlite3.OperationalError("simulated write failure")
        failure.setattr(storage, "update_user_in", fail)
        with pytest.raises(sqlite3.OperationalError):
            discounts.settle(USER, paid_event(checkout), NOW)
    assert not billing.history(USER)
    with sqlite3.connect(storage.db_path()) as reopened:
        assert reopened.execute("SELECT percent FROM pro_discounts WHERE status = 'active'").fetchone() == (50,)
        assert reopened.execute("SELECT query_id FROM pro_discount_orders").fetchone() == ("q",)
    assert discounts.settle(USER, paid_event(checkout), NOW)[0] == "paid"
