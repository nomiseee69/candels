import hashlib
import hmac
import io
import json
import time
from urllib.parse import urlencode

import pytest

import billing
import discounts
import mini_app
import mini_star_audit
import notifier
import storage

TOKEN = "test-token"
BUYER = 8123


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr(notifier, "TOKEN", TOKEN)
    monkeypatch.setenv("ADMIN_IDS", "1")


def signed(buyer=BUYER, time_offset=0):
    fields = {"auth_date": str(int(time.time()) + time_offset), "user": json.dumps({"id": buyer}),
              "signature": "test-signature"}
    data = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    key = hmac.digest(b"WebAppData", TOKEN.encode(), hashlib.sha256)
    return urlencode({**fields, "hash": hmac.new(key, data.encode(), hashlib.sha256).hexdigest()})


def invoke(path, data, auth):
    raw = json.dumps(data).encode()
    statuses = []
    result = mini_app.application({
        "PATH_INFO": path, "REQUEST_METHOD": "POST", "HTTP_X_TELEGRAM_INIT_DATA": auth,
        "CONTENT_LENGTH": str(len(raw)), "wsgi.input": io.BytesIO(raw),
    }, lambda status, headers: statuses.append(status))
    return statuses[0], json.loads(b"".join(result))


def test_authorization_and_isolation():
    valid = signed()
    assert invoke("/api/state", {}, valid)[0] == "200 OK"
    for forged in (valid.replace(str(BUYER), "99999"), signed(time_offset=-901),
                   signed(time_offset=60), valid + "&user=test", ""):
        assert invoke("/api/state", {}, forged)[0] == "401 Unauthorized"
    billing.extend(BUYER, 30)
    assert invoke("/api/state", {"user_id": 99999}, valid)[1]["pro"]
    assert not invoke("/api/state", {"user_id": BUYER}, signed(99999))[1]["pro"]
    assert not storage.get_user(99999)


def mock_invoice(method, data):
    assert method == "createInvoiceLink" and data["currency"] == "XTR"
    return {"ok": True, "result": "https://t.me/$test"}


@pytest.mark.parametrize("plan", ["month", "year"])
def test_invoice_binding_settlement_and_refund(monkeypatch, plan):
    calls = []

    def api(method, data):
        calls.append(data)
        return mock_invoice(method, data)

    monkeypatch.setattr(notifier, "api_post", api)
    _, issued = invoke("/api/invoice", {"plan": plan, "discounted": False, "amount": 1}, signed())
    assert issued["url"] == "https://t.me/$test"
    data, = calls
    assert (data.get("subscription_period") == billing.MONTH_SECONDS) == (plan == "month")
    price = billing.PLANS[plan][1]
    assert json.loads(data["prices"])[0]["amount"] == price
    event = {"id": "checkout", "invoice_payload": issued["payload"], "currency": "XTR",
             "total_amount": price, "from": {"id": BUYER}, "telegram_payment_charge_id": "paid"}
    now = int(time.time())
    if plan == "month":
        event.update(is_recurring=True, is_first_recurring=True, subscription_expiration_date=now + 30 * billing.DAY)
    assert not billing.check_pre_checkout({**event, "from": {"id": BUYER + 1}})
    assert not billing.check_pre_checkout({**event, "total_amount": 1})
    assert billing.check_pre_checkout(event)
    assert not mini_app.state(BUYER, issued["payload"])["paid"]
    wrong = {**event, "telegram_payment_charge_id": "wrong-user"}
    assert "поддержка" in billing.stars_paid(BUYER + 1, wrong)
    assert not mini_app.state(BUYER + 1)["pro"]
    assert billing.stars_paid(BUYER, event, now)
    assert billing.stars_paid(BUYER, event, now) is None
    assert mini_app.state(BUYER, issued["payload"])["paid"]
    assert not mini_app.state(BUYER + 1, issued["payload"])["paid"]
    assert not billing.stars_refunded(BUYER + 1, event)
    assert billing.stars_refunded(BUYER, event)
    assert not mini_app.state(BUYER, issued["payload"])["pro"]
    if plan == "month":
        renewal = {**event, "telegram_payment_charge_id": "renewed", "is_first_recurring": False,
                   "subscription_expiration_date": now + 60 * billing.DAY}
        assert billing.stars_paid(BUYER, renewal, now + 30 * billing.DAY)
        assert billing.status(BUYER, now)["until"] == now + 60 * billing.DAY


def test_discount_order_and_official_audit():
    billing.status(BUYER)
    assert discounts.issue(1, BUYER, 50, None) == "issued"
    invoice = mini_app.invoice(BUYER, "month", True, mock_invoice)
    amount = discounts.price(billing.PLANS["month"][1], 50)
    payment = {"invoice_payload": invoice["payload"], "currency": "XTR", "total_amount": amount,
               "telegram_payment_charge_id": "discount", "id": "q", "from": {"id": BUYER}}
    assert discounts.pre_checkout(payment) == "ok"
    assert discounts.settle(BUYER, payment)[0] == "paid"
    assert discounts.settle(BUYER, payment)[0] == "duplicate"
    assert not mini_app.state(BUYER)["recurring"]
    peer = {"type": "user", "user": {"id": BUYER}, "transaction_type": "invoice_payment",
            "affiliate": {"amount": 25, "commission_per_mille": 100}}
    snapshot = [{"id": "discount", "date": 42, "amount": 225, "source": peer},
                {"id": "discount", "date": 48, "amount": 225,
                 "receiver": {**peer, "affiliate": {"amount": -25, "commission_per_mille": 100}}}]
    assert mini_star_audit.reconcile(snapshot) == 2
    assert mini_star_audit.reconcile(snapshot) == 0
    assert mini_star_audit.summary() == [("payment", 1, amount, 225 * 10**9, 1, 25 * 10**9),
                                        ("refund", 1, amount, -225 * 10**9, 1, -25 * 10**9)]
    with storage.transaction() as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'referral_%'").fetchall()


def test_shared_gate_trial_parallel_and_cancel():
    from concurrent.futures import ThreadPoolExecutor
    import mini_orders
    import subscription_guard as guard

    now = int(time.time())
    billing.status(BUYER, now)
    with ThreadPoolExecutor(2) as pool:
        quotes = list(pool.map(lambda _: mini_app.invoice(BUYER, "month", False, mock_invoice), range(2)))
    assert quotes[0]["payload"] == quotes[1]["payload"]
    native = {"id": "native", "from": {"id": BUYER}, "invoice_payload": quotes[0]["payload"],
              "currency": "XTR", "total_amount": 500}
    legacy = {**native, "id": "legacy", "invoice_payload": "pro_month_500"}
    assert billing.check_pre_checkout(native)
    assert not billing.check_pre_checkout(legacy)
    assert mini_orders.checkout(native, now + mini_orders.TTL + 5) is False
    payment = {**native, "telegram_payment_charge_id": "initial", "is_first_recurring": True,
               "is_recurring": True, "subscription_expiration_date": now + 30 * billing.DAY}
    assert billing.stars_paid(BUYER, payment, now)
    assert not billing.check_pre_checkout(legacy)
    with pytest.raises(ValueError, match="уже оплачен"):
        mini_app.invoice(BUYER, "year", False, mock_invoice)
    calls = []
    for _ in range(2):
        mini_app.cancel_renewal(BUYER, lambda method, data: calls.append((method, data)) or {"ok": True})
    assert calls == [("editUserStarSubscription", {
        "user_id": BUYER, "telegram_payment_charge_id": "initial", "is_canceled": True})]
    assert mini_app.state(BUYER)["cancelled"]
    assert billing.status(BUYER)["until"] == payment["subscription_expiration_date"]
    extra = {**legacy, "telegram_payment_charge_id": "extra", "is_first_recurring": True,
             "subscription_expiration_date": payment["subscription_expiration_date"]}
    assert "возврат" in billing.stars_paid(BUYER, extra, now)
    guard.process_conflicts(lambda method, data: calls.append((method, data)) or {"ok": True})
    assert [method for method, _ in calls] == ["editUserStarSubscription", "editUserStarSubscription", "refundStarPayment"]
    assert billing.status(BUYER)["until"] == payment["subscription_expiration_date"]
    assert mini_orders.create(BUYER, "year", 4800, now + 31 * billing.DAY)


def test_existing_schema_preserves_settings_and_shared_database(tmp_path):
    import os
    import subprocess
    import sys

    with storage.transaction() as db:
        db.execute("INSERT INTO users VALUES (?,?)", (str(BUYER), '{"settings":{"coins":42},"patterns":[]}'))
    signed_invoice = mini_app.invoice(BUYER, "year", False, mock_invoice)
    script = ("import sys, storage, mini_orders; storage.FILE=sys.argv[1]; "
              "assert mini_orders.create(8123,'year',4800)==sys.argv[2]")
    subprocess.run([sys.executable, "-c", script, storage.FILE, signed_invoice["payload"]],
                   env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"}, check=True)
    script = ("import sys,time,storage; storage.FILE=sys.argv[1]; "
              "db=storage._connect(); db.execute('BEGIN IMMEDIATE'); "
              "print('locked',flush=True); time.sleep(.5); db.commit(); db.close()")
    with subprocess.Popen([sys.executable, "-c", script, storage.FILE],
                          stdout=subprocess.PIPE, text=True) as worker:
        assert worker.stdout.readline().strip() == "locked"
        assert mini_app.invoice(BUYER, "year", False, mock_invoice)["payload"] == signed_invoice["payload"]
        assert worker.wait(timeout=5) == 0
    assert storage.get_user(BUYER)["settings"] == {"coins": 42}
