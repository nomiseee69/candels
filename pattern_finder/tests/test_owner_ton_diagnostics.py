import base64
from pathlib import Path

import pytest
from pytoniq_core import Address, StateInit
from pytoniq_core.boc.builder import Builder

import notifier
import owner_ton_diagnostics as owner
import partner_finance as finance
import partner_ton as ton
import storage
from test_mini_app import TOKEN, invoke, signed
from test_partners import NOW, proof_fixture


DOMAIN = "findercandles.duckdns.org"
OTHER = "0:" + "ab"*32


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr(notifier, "TOKEN", TOKEN)
    monkeypatch.setenv("ADMIN_IDS", "1")
    monkeypatch.setenv("REFERRALS_ENABLED", "0")
    monkeypatch.setenv("REFERRAL_PAYOUTS_ENABLED", "0")
    monkeypatch.setenv("TON_OWNER_WALLET_DIAGNOSTICS", "1")
    monkeypatch.setenv("TON_OWNER_INCOME_ADDRESS", Address(OTHER).to_str())
    monkeypatch.setenv("TON_OWNER_PARTNER_ADDRESS", Address("0:" + "de"*32).to_str())
    monkeypatch.setenv("TON_CONNECT_APP_DOMAIN", DOMAIN)
    monkeypatch.delenv("TON_CONNECT_MANIFEST_URL", raising=False)
    monkeypatch.setattr(finance, "timestamp", lambda now=None: NOW if now is None else int(now))

    def offline(path):
        raise AssertionError("Unexpected external call")

    monkeypatch.setattr(ton, "get", offline)


def api(route, data=None, user=1):
    return invoke("/api/owner-wallets/"+route, data or {}, signed(user) if user else "")


def test_authorization_and_feature_flag(monkeypatch):
    assert api("state", user=None)[0] == "401 Unauthorized"
    for route in ("state", "challenge", "verify"):
        assert api(route, {"slot": "income", "user_id": 1}, user=7)[0] == "404 Not Found"
    monkeypatch.setenv("TON_OWNER_WALLET_DIAGNOSTICS", "0")
    assert api("state")[0] == "404 Not Found"
    assert not Path(storage.db_path()).with_name("owner_ton_diagnostics.db").exists()
    monkeypatch.setenv("TON_OWNER_WALLET_DIAGNOSTICS", "1")
    assert api("state")[1]["cards"]["income"]["checksum"]
    assert not Path(storage.db_path()).exists()


def test_address_checksum_and_workchain(monkeypatch):
    address = Address(OTHER).to_str()
    for invalid in (address[:-1] + ("A" if address[-1] != "A" else "B"), Address(OTHER).to_str(is_test_only=True),
                    Address("-1:" + "11"*32).to_str()):
        monkeypatch.setenv("TON_OWNER_INCOME_ADDRESS", invalid)
        assert api("state")[1]["cards"]["income"]["checksum"] is False


def test_wallet_version_hashes_and_v5_library_reference():
    layouts = {owner.VERSIONS[code]: offset for code, offset in ton.WALLET_CODES.items()}
    assert layouts == {"V4R2": 64, "V5R1": 65}
    code = next(code for code, version in owner.VERSIONS.items() if version == "V5R1")
    library = Builder(type_=2).store_uint(2, 8).store_bytes(bytes.fromhex(code)).end_cell()
    assert ton.wallet_code_hash(ton.Cell.one_from_boc(library.to_boc())) == code


def prepare(monkeypatch, account_state="active", *, missing_jetton=False, fail_provider=False):
    account, _, _ = proof_fixture(monkeypatch)
    address = account["address"]
    init = StateInit.deserialize(ton.Cell.one_from_boc(base64.b64decode(account["walletStateInit"])).begin_parse())
    monkeypatch.setitem(owner.VERSIONS, init.code.hash.hex(), "V4R2")
    monkeypatch.setenv("TON_OWNER_INCOME_ADDRESS", Address(address).to_str(is_bounceable=False))

    def provider(path):
        if fail_provider:
            raise finance.Error("offline")
        if path == f"blockchain/accounts/{address}":
            return {"address": address, "status": account_state, "balance": 1250000000,
                    "code": init.code.to_boc().hex() if account_state == "active" else None}
        if missing_jetton:
            raise ton.NotFound("not indexed")
        return {"jetton": {"address": ton.MASTER, "decimals": 6}, "balance": "2510000",
                "wallet_address": {"address": OTHER}}

    monkeypatch.setattr(ton, "get", provider)
    if fail_provider or account_state != "active":
        def no_key(_):
            raise finance.Error("offline") if fail_provider else ton.NotFound("not indexed")
        # proof_fixture patches wallet_key again when signing. The metadata layer stays independent here.
        monkeypatch.setattr(owner, "_key", no_key)
    status, body = api("challenge", {"slot": "income"})
    assert status == "200 OK"
    account, proof, _ = proof_fixture(monkeypatch, body["payload"])
    return {"slot": "income", "account": account, "proof": proof}, body["report"]


def test_valid_proof_and_no_financial_writes(monkeypatch):
    tables = ("fc_wallets", "fc_treasury_ledger", "fc_referral_ledger", "fc_payout_requests")
    with storage.transaction() as db:
        finance.schema(db)
        before = {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
    data, report = prepare(monkeypatch)
    assert report["signature"] == "pending"
    status, body = api("verify", data)
    assert status == "200 OK"
    report = body["report"]
    assert (report["mainnet"], report["signature"], report["ton"], report["usdt"]) == ("confirmed", "confirmed", "1.25", "2.51")
    assert api("verify", data)[0] == "400 Bad Request"
    assert api("state")[1]["cards"]["income"]["report"] == report
    assert api("state")[1]["cards"]["partner"]["report"] is None
    with storage.transaction() as db:
        assert {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables} == before
    with owner.transaction() as db:
        assert data["proof"]["signature"] not in "".join(db.iterdump())


def test_admin_can_open_diagnostics_without_pro_profile(monkeypatch):
    import bot

    monkeypatch.setattr(bot, "stars_balance", lambda: None)
    monkeypatch.setattr(bot.billing, "stats", lambda: "")
    monkeypatch.setenv("MINI_APP_PROFILE_ENABLED", "0")
    monkeypatch.setenv("MINI_APP_URL", "https://findercandles.duckdns.org/")
    def web_buttons(user):
        return [item for row in bot.admin_view(user)[1]["inline_keyboard"] for item in row if "web_app" in item]
    assert len(web_buttons(1)) == 1
    assert web_buttons(7) == []


@pytest.mark.parametrize("problem", ["domain", "network", "address", "signature", "contract"])
def test_bad_proofs_are_rejected_and_consumed(monkeypatch, problem):
    data, _ = prepare(monkeypatch)
    if problem == "domain":
        data["proof"]["domain"] = {"value": "invalid.example", "lengthBytes": 15}
    elif problem == "network":
        data["account"]["chain"] = "-3"
    elif problem == "address":
        data["account"]["address"] = OTHER
    elif problem == "signature":
        data["proof"]["signature"] = base64.b64encode(b"\0"*64).decode()
    else:
        monkeypatch.setattr(ton, "WALLET_CODES", {})
    status, body = api("verify", data)
    assert status == "200 OK" and body["report"]["mainnet"] != "confirmed"
    assert body["report"]["signature"] == ("unsupported" if problem == "contract" else "failed")
    assert api("verify", data)[0] == "400 Bad Request"


def test_expiry_binding_and_rate_limit(monkeypatch):
    data, _ = prepare(monkeypatch)
    assert api("verify", {**data, "slot": "partner"})[0] == "400 Bad Request"
    monkeypatch.setattr(finance, "timestamp", lambda: NOW+600)
    assert api("verify", data)[0] == "400 Bad Request"
    for _ in range(4):
        assert api("challenge", {"slot": "income"})[0] == "200 OK"
    assert api("challenge", {"slot": "income"})[0] == "429 Too Many Requests"


@pytest.mark.parametrize("state, outage", [("active", False), ("uninit", False), ("active", True)])
def test_missing_jetton_and_outage_are_separate_from_signature(monkeypatch, state, outage):
    data, _ = prepare(monkeypatch, state, missing_jetton=True, fail_provider=outage)
    report = api("verify", data)[1]["report"]
    assert report["signature"] == "confirmed"
    assert report["jetton"] == ("unavailable" if outage else "not_found")
    assert report["usdt"] is None
    assert report["mainnet"] == ("confirmed" if state == "active" and not outage else "pending")
