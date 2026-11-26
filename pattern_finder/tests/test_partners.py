import base64
from concurrent.futures import ThreadPoolExecutor
import time

import pytest
from nacl.signing import SigningKey
from pytoniq_core import Address, StateInit, begin_cell
from pytoniq_core.boc.hashmap.hashmap import HashMap

import billing
import discounts
import mini_star_audit
import partner_admin
import partner_finance as f
import partner_payouts as payouts
import partner_ton as ton
import storage

NOW = int(time.time())
OWNER = "0:" + "ab" * 32
SOURCE = "0:" + "bc" * 32
WALLET = "0:" + "cd" * 32
WALLET2 = "0:" + "de" * 32
PARTNER, BUYER = 101, 301
DAY = f.DAY


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setenv("REFERRALS_ENABLED", "1")
    monkeypatch.setenv("REFERRAL_PAYOUTS_ENABLED", "1")
    monkeypatch.setenv("REFERRAL_NATIVE_ENDED_AT", str(NOW - 100*DAY))
    monkeypatch.setenv("MINI_APP_AFFILIATE_ACTIVE", "0")
    monkeypatch.setenv("TON_TREASURY_ADDRESS", OWNER)
    monkeypatch.setenv("TON_PAYOUT_SOURCE_ADDRESS", OWNER)
    monkeypatch.setenv("TON_SETTLEMENT_SOURCE_ADDRESS", SOURCE)
    monkeypatch.setenv("ADMIN_IDS", "1")


def code(user=PARTNER):
    return f.link(user).split("fc_")[1]


def invite(buyer=BUYER, now=NOW):
    assert f.attribute(buyer, code(), now)


def invoice(charge="first", *, plan="month", stars=500, **fields):
    return {"telegram_payment_charge_id": charge, "invoice_payload": f"pro_{plan}_{stars}",
            "total_amount": stars, "currency": "XTR", **fields}


def balance():
    with storage.transaction() as db:
        return f.balances(db, PARTNER)


def test_attribution_permanent_concurrent_and_safe_cutover(monkeypatch):
    first, second = code(), code(102)
    assert len(first) == 32 and first == code()
    assert not f.attribute(PARTNER, first) and not f.attribute(102, first)
    storage.update_user(300, lambda user: user.update(welcomed=True))
    assert not f.attribute(300, first)
    with ThreadPoolExecutor(2) as pool:
        decisions = list(pool.map(lambda c: f.attribute(BUYER, c), [first, second]))
    assert sum(decisions) == 1
    assert not f.attribute(BUYER, first) and not f.attribute(BUYER, second)
    monkeypatch.setenv("MINI_APP_AFFILIATE_ACTIVE", "1")
    assert not f.attribute(999, first)


@pytest.mark.parametrize("plan,stars", [("month", 500), ("year", 4800)])
def test_first_purchase_refunds_and_duplicate_atomicity(plan, stars, monkeypatch):
    invite(now=NOW-25*DAY)
    start = NOW-24*DAY
    first = invoice(plan=plan, stars=stars)
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(lambda _: billing.stars_paid(BUYER, first, start), range(2)))
    assert sum(bool(text) for text in outcomes) == 1
    assert payouts.overview(PARTNER, start)["counts"] == {"pending": 1}
    assert payouts.overview(PARTNER)["counts"] == {"matured_unfunded": 1}
    assert payouts.overview(PARTNER)["available"] == "0"
    assert billing.stars_paid(BUYER, invoice("another"), NOW+400*DAY)
    assert billing.stars_paid(BUYER, invoice("renew", is_recurring=True), NOW+500*DAY)
    assert billing.stars_refunded(BUYER, first)
    assert payouts.overview(PARTNER)["counts"] == {"reversed": 1}


def test_disabled_test_payment_and_invalid_currency(monkeypatch):
    invite()
    assert billing.stars_paid(BUYER, {**invoice(), "currency": "USD"}) is None
    monkeypatch.setenv("REFERRAL_TEST_MODE", "1")
    assert billing.stars_paid(BUYER, invoice(test=True))
    assert not payouts.overview(PARTNER)["purchases"]


def test_discount_actual_amount_and_trial():
    invite(now=NOW-60)
    billing.status(BUYER, NOW)
    assert discounts.issue(1, BUYER, 50, None, NOW) == "issued"
    quote = discounts.invoice(BUYER, "month", NOW)
    event = {**invoice(stars=250), "invoice_payload": quote["payload"], "id": "q", "from": {"id": BUYER}}
    assert discounts.pre_checkout(event, NOW) == "ok"
    assert discounts.settle(BUYER, event, NOW)[0] == "paid"
    assert discounts.settle(BUYER, event, NOW)[0] == "duplicate"
    with storage.transaction() as db:
        assert db.execute("SELECT stars,discount FROM fc_referral_commissions").fetchone() == (250, 50)


def fund(monkeypatch, proceeds=100_000_001):
    invite(now=NOW-25*DAY)
    payment = invoice(stars=500)
    assert billing.stars_paid(BUYER, payment, NOW-24*DAY)
    ledger = {"id":"first", "date":NOW-24*DAY, "amount":500,
              "source":{"type":"user","transaction_type":"invoice_payment","user":{"id":BUYER}}}
    fragment = {"id":"fragment-ref", "date":NOW-2*DAY, "amount":500,
                "receiver":{"type":"fragment","withdrawal_state":{"type":"succeeded","date":NOW-2*DAY}}}
    mini_star_audit.reconcile([ledger])
    topup = ton.Transfer("1"*64, "2"*64, SOURCE, OWNER, proceeds, NOW-1)
    monkeypatch.setattr(ton, "verify_transfer", lambda *a, **k: topup)
    monkeypatch.setattr(ton, "jetton_wallet", lambda owner: (owner, proceeds))
    with pytest.raises(f.Error, match="Fragment"):
        payouts.settle_batch(1, topup.root, "fragment-ref", 500, 10**9, 0, ["first"], NOW)
    assert payouts.overview(PARTNER)["available"] == "0"
    mini_star_audit.reconcile([fragment])
    assert payouts.settle_batch(1, topup.root, "fragment-ref", 500, 10**9, 0, ["first"], NOW)
    assert payouts.overview(PARTNER)["available"] == f.display(proceeds//10)
    with pytest.raises(f.Error, match="уже учтены"):
        payouts.settle_batch(1, topup.root, "fragment-ref", 500, 10**9, 0, ["first"], NOW)
    with storage.transaction() as db:
        db.execute("INSERT INTO fc_wallets VALUES (?,?,?,?,1,'TON_MAINNET')", (PARTNER, WALLET, NOW, NOW))
    return payment


def test_funded_rounding_reservation_concurrency_and_refund_debt(monkeypatch):
    payment = fund(monkeypatch)
    assert balance() == (10_000_000, 0, 0, 0)
    with pytest.raises(f.Error, match="минимальной"):
        payouts.request(PARTNER, f.amount("9.99"))
    with pytest.raises(f.Error, match="Недостаточно"):
        payouts.request(PARTNER, f.amount("10.01"))

    def attempt(_):
        try:
            return payouts.request(PARTNER, f.amount("10.00"), now=NOW)
        except f.Error:
            return None

    with ThreadPoolExecutor(2) as pool:
        key, = [result for result in pool.map(attempt, range(2)) if result]
    assert balance() == (0, 10_000_000, 0, 0)
    payouts.decision(1, key, True, "Проверено", NOW)
    payouts.dispatch(1, key, "Отправка владельцем", NOW)
    with pytest.raises(f.Error):
        payouts.decision(1, key, False, "Попытка отмены", NOW)
    tx = ton.Transfer("3"*64, "4"*64, OWNER, WALLET, 10_000_000, NOW)
    payouts.submit(1, key, tx.root, NOW)
    monkeypatch.setattr(ton, "verify_transfer", lambda *a: tx)
    payouts.verify_payout(1, key, NOW)
    with pytest.raises(f.Error):
        payouts.verify_payout(1, key, NOW)
    assert billing.stars_refunded(BUYER, payment)
    assert balance() == (0, 0, 10_000_000, 10_000_000)


def test_insufficient_reserve_and_refund_before_dispatch(monkeypatch):
    payment = fund(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(ton, "jetton_wallet", lambda owner: (owner, 9_999_999))
        assert payouts.overview(PARTNER)["available"] == "0"
        with pytest.raises(f.Error, match="резерва"):
            payouts.request(PARTNER, 10_000_000)
    key = payouts.request(PARTNER, 10_000_000)
    assert billing.stars_refunded(BUYER, payment)
    assert balance() == (0, 0, 0, 0)
    with storage.transaction() as db:
        assert db.execute("SELECT status FROM fc_payout_requests WHERE id=?", (key,)).fetchone() == ("cancelled",)


def test_quote_history_does_not_create_available_balance(monkeypatch):
    invite()
    with pytest.raises(f.Error):
        f.set_rate(1, -10, "отчёт")
    f.set_rate(1, 12_000, "первая выписка")
    assert billing.stars_paid(BUYER, invoice())
    f.set_rate(1, 14_000, "следующая выписка")
    with storage.transaction() as db:
        assert db.execute("SELECT estimate FROM fc_referral_commissions").fetchone() == (600_000,)
        assert db.execute("SELECT COUNT(*) FROM fc_rate_history").fetchone() == (2,)
    assert balance() == (0, 0, 0, 0)


def proof_fixture(monkeypatch, token="test-challenge"):
    # RFC 8032 public test vector. This key has never belonged to a real wallet.
    signing = SigningKey(bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc4"
                                            "4449c5697b326919703bac031cae7f60"))
    key = bytes(signing.verify_key)
    code_cell = begin_cell().store_uint(15, 16).end_cell()
    data = begin_cell().store_uint(0, 64).store_bytes(key).end_cell()
    init = StateInit(code=code_cell, data=data).serialize()
    owner = "0:" + init.hash.hex()
    domain = "findercandles.duckdns.org"
    proof = {"timestamp":NOW, "domain":{"value":domain,"lengthBytes":len(domain)}, "payload":token,
             "signature":base64.b64encode(signing.sign(ton.proof_digest(owner,domain,NOW,token)).signature).decode()}
    account = {"address":owner, "publicKey":key.hex(), "walletStateInit":base64.b64encode(init.to_boc()).decode(),
               "chain":"-239"}
    monkeypatch.setattr(ton, "wallet_key", lambda address: key)
    monkeypatch.setitem(ton.WALLET_CODES, code_cell.hash.hex(), 64)
    return account, proof, domain


def test_ton_proof_valid_bad_inputs_and_challenge_replay(monkeypatch):
    code()
    token = payouts.challenge(PARTNER, NOW)
    account, proof, domain = proof_fixture(monkeypatch, token)
    for a, p, d in [
        ({**account,"chain":"-3"},proof,domain), ({**account,"address":WALLET},proof,domain),
        (account,{**proof,"signature":"AA=="},domain), (account,{**proof,"timestamp":NOW-601},domain),
        (account,proof,"other.example"), ({**account,"publicKey":"af"*32},proof,domain),
    ]:
        with pytest.raises(f.Error):
            payouts.bind(PARTNER, a, p, d, NOW)
    with monkeypatch.context() as patch:
        patch.setattr(ton, "WALLET_CODES", {})
        with pytest.raises(f.Error):
            payouts.bind(PARTNER, account, proof, domain, NOW)
    payouts.bind(PARTNER, account, proof, domain, NOW)
    with pytest.raises(f.Error):
        payouts.bind(PARTNER, account, proof, domain, NOW)
    # Disconnecting cannot bypass the cooldown for replacing this address.
    payouts.disconnect(PARTNER)
    token = payouts.challenge(PARTNER, NOW)
    monkeypatch.setattr(ton,"proof_owner",lambda *args: WALLET2)
    payouts.bind(PARTNER, account, {**proof,"payload":token}, domain, NOW)
    assert payouts.overview(PARTNER)["wallet"]["unlock"] == NOW + 48*3600


def test_verify_official_usdt_trace_and_reject_bounce_or_missing_child(monkeypatch):
    src_wallet, dst_wallet = "0:"+"11"*32, "0:"+"22"*32
    root, credit_tx, edge1, edge2 = ("3"*64, "4"*64, "5"*64, "6"*64)
    body = (begin_cell().store_uint(0x178D4519,32).store_uint(0,64).store_coins(10_000_000)
            .store_address(Address(OWNER)).end_cell().to_boc().hex())

    def tx(identifier, account, in_msg, *out):
        return {"hash":identifier,"account":{"address":account},"success":True,"aborted":False,
                "destroyed":False,"in_msg":{"bounced":False,**in_msg},"out_msgs":list(out),"utime":NOW}
    child = {"transaction":tx(credit_tx,dst_wallet,{"hash":edge2,"source":{"address":src_wallet},"raw_body":body})}
    middle = {"transaction":tx("7"*64,src_wallet,{"hash":edge1},{"hash":edge2,"msg_type":"int_msg"}),
              "children":[child]}
    trace = {"transaction":tx(root, OWNER, {}, {"hash":edge1,"msg_type":"int_msg"}),"children":[middle]}
    monkeypatch.setattr(ton,"get",lambda path: trace)
    monkeypatch.setattr(ton,"jetton_wallet",lambda owner:(src_wallet if owner==OWNER else
                       dst_wallet if owner==WALLET else WALLET2, 10_000_000))
    assert ton.verify_transfer(credit_tx,OWNER,WALLET,10_000_000).root == root
    for args in [("8"*64,OWNER,WALLET,10_000_000),(root,WALLET,OWNER,10_000_000),
                 (root,OWNER,WALLET2,10_000_000),(root,OWNER,WALLET,1)]:
        with pytest.raises(f.Error):
            ton.verify_transfer(*args)
    child["transaction"]["in_msg"]["bounced"] = True
    with pytest.raises(f.Error):
        ton.verify_transfer(root,OWNER,WALLET,10_000_000)
    middle["children"] = []
    with pytest.raises(f.Error, match="Индексатор"):
        ton.verify_transfer(root,OWNER,WALLET,10_000_000)


def test_ton_trace_retains_raw_messages_when_json_list_is_empty():
    internal = begin_cell().store_bit(0).end_cell()
    external = begin_cell().store_bits("11").end_cell()
    dictionary = HashMap(15, map_={0: internal, 1: external},
                         value_serializer=lambda msg, out: out.store_ref(msg)).serialize()
    messages = begin_cell().store_bit(0).store_dict(dictionary).end_cell()
    cell = (begin_cell().store_uint(0b0111, 4).store_uint(0, 672).store_uint(2, 15)
            .store_ref(messages).end_cell())
    transaction = {"hash": cell.hash.hex(), "raw": cell.to_boc().hex(), "out_msgs": []}
    child_tx = {"in_msg": {"hash": internal.hash.hex()}, "out_msgs": []}
    trace = {"transaction": transaction, "children": [{"transaction": child_tx}]}
    assert list(ton._transactions(trace)) == [(transaction, [internal.hash.hex()]), (child_tx, [])]
    trace["children"] = []
    with pytest.raises(f.Error, match="Индексатор"):
        list(ton._transactions(trace))


def test_reject_wrong_master_and_decimals(monkeypatch):
    records = {"wallet_address":{"address":WALLET}, "balance":"10000000",
               "jetton":{"address":ton.MASTER, "decimals":9}}
    monkeypatch.setattr(ton,"get",lambda path: records)
    with pytest.raises(f.Error):
        ton.jetton_wallet(OWNER)
    records["jetton"] = {"address":WALLET2,"decimals":6}
    with pytest.raises(f.Error):
        ton.jetton_wallet(OWNER)


def test_manifest_and_partner_routes_require_signed_user(monkeypatch):
    from test_mini_app import invoke, signed, BUYER as WEB_USER
    import mini_app
    import notifier
    import json

    monkeypatch.setattr(notifier, "TOKEN", "test-token")
    metadata = []
    response = mini_app.application({"PATH_INFO": "/tonconnect-manifest.json", "REQUEST_METHOD": "GET"},
                                    lambda status, headers: metadata.append((status, dict(headers))))
    status, headers = metadata[0]
    assert status == "200 OK" and headers["Access-Control-Allow-Origin"] == "*"
    manifest = json.loads(b"".join(response))
    assert manifest["iconUrl"] == "https://findercandles.duckdns.org/tonconnect-icon.png"
    assert invoke("/api/partners/register", {}, "")[0] == "401 Unauthorized"
    first = invoke("/api/partners/register", {"user_id": PARTNER}, signed())[1]["link"]
    second = invoke("/api/partners/register", {}, signed(PARTNER))[1]["link"]
    assert first != second and invoke("/api/partners/state", {}, signed(WEB_USER))[1]["link"] == first


def test_ton_outage_does_not_spend_reserve(monkeypatch):
    payment = fund(monkeypatch)
    key = payouts.request(PARTNER, 10_000_000, now=NOW)
    payouts.decision(1, key, True, "Проверено", NOW)
    payouts.dispatch(1, key, "Передано на отправку", NOW)
    payouts.submit(1, key, "3"*64, NOW)

    def outage(*args):
        raise f.Error("Нет ответа индексатора")

    monkeypatch.setattr(ton, "verify_transfer", outage)
    with pytest.raises(f.Error, match="индексатора"):
        payouts.verify_payout(1, key, NOW)
    assert balance() == (0, 10_000_000, 0, 0)
    assert billing.stars_refunded(BUYER, payment)
    assert balance() == (0, 10_000_000, 0, 10_000_000)
    with storage.transaction() as db:
        assert db.execute("SELECT status FROM fc_payout_requests").fetchone() == ("submitted",)


def test_ledger_is_append_only(monkeypatch):
    import sqlite3

    fund(monkeypatch)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), storage.transaction() as db:
        db.execute("DELETE FROM fc_referral_ledger")
    assert balance() == (10_000_000, 0, 0, 0)


def test_above_minimum_and_funding_failure_are_atomic(monkeypatch):
    fund(monkeypatch, 120_000_000)

    def fail_reserve(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(f, "journal", fail_reserve)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            payouts.request(PARTNER, f.amount("10.25"), now=NOW)
    assert balance() == (12_000_000, 0, 0, 0)
    key = payouts.request(PARTNER, f.amount("10.25"), now=NOW)
    assert payouts.overview(PARTNER)["payouts"][0][:4] == (key, "10250000", "requested", NOW+7*DAY)
    assert balance() == (1_750_000, 10_250_000, 0, 0)
    assert f"№{key} · партнёр {PARTNER} · 10.25 USDT · requested" in partner_admin.command(1, f"payout {key}")
    assert "Fragment: 500 Stars" in partner_admin.command(1, "reconciliation")


def test_notifications_retry_after_telegram_outage(monkeypatch):
    invite()
    monkeypatch.setattr(f.time, "time", lambda: NOW)
    attempts = []

    def offline(method, data):
        attempts.append((method, data["chat_id"]))
        return {"ok": False}

    f.send_notifications(offline)
    f.send_notifications(offline)
    assert attempts == [("sendMessage", PARTNER)]
    monkeypatch.setattr(f.time, "time", lambda: NOW+3601)
    f.send_notifications(lambda method, data: {"ok": True})
    with storage.transaction() as db:
        assert db.execute("SELECT sent FROM fc_notifications").fetchone() == (NOW+3601,)
