"""Owner-only wallet diagnostics with an isolated SQLite store and read-only TON calls."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3

from pytoniq_core import Cell
from pytoniq_core.boc.deserialize import BocError

import billing
import partner_finance as finance
import partner_ton as ton
import storage


SLOTS = {"income": "TON_OWNER_INCOME_ADDRESS", "partner": "TON_OWNER_PARTNER_ADDRESS"}
VERSIONS = {
    "b61041a58a7980b946e8fb9e198e3c904d24799ffa36574ea4251c41a566f581": "V3R1",
    "84dafa449f98a6987789ba232358072bc0f76dc4524002a5d0918b9a75d2d599": "V3R2",
    "64dd54805522c5be8a9db59cea0105ccf0d08786ca79beb8cb79e880a8d7322d": "V4R1",
    "feb5ff6820e2ff0d9483e7e0d62c817d846789fb4ae580c878866d959dabd5c0": "V4R2",
    "20834b7b72b112147e1b2fb457b84e74d1a30f04f737d4f62a668e9552d2b72f": "V5R1",
    "9149ae51c1e4689710cebf7830297b16acfbadb363a920a537893e7ffeeca768": "WalletTg",
}


class RateLimited(finance.Error):
    pass


def allowed(admin):
    return finance.flag("TON_OWNER_WALLET_DIAGNOSTICS") and admin in billing.admins()


@contextmanager
def transaction():
    path = Path(storage.db_path()).with_name("owner_ton_diagnostics.db")
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE IF NOT EXISTS challenges (admin INTEGER, slot TEXT, address TEXT, domain TEXT,
            digest TEXT, issued INTEGER, PRIMARY KEY(admin,slot))""")
        db.execute("""CREATE TABLE IF NOT EXISTS results (admin INTEGER, slot TEXT, address TEXT, domain TEXT,
            report TEXT, PRIMARY KEY(admin,slot))""")
        db.execute("CREATE TABLE IF NOT EXISTS limits (scope TEXT PRIMARY KEY, expires INTEGER, hits INTEGER)")
        yield db


def limit(admin, action, now):
    scopes = [(f"{admin}:{action}", 12 if action == "state" else 4)]
    if action != "state":
        scopes.append(("ton", 12))
    with transaction() as db:
        db.execute("DELETE FROM limits WHERE expires<=?", (now,))
        for scope, maximum in scopes:
            row = db.execute("SELECT hits FROM limits WHERE scope=?", (scope,)).fetchone()
            if row and row[0] >= maximum:
                raise RateLimited("Слишком много запросов. Повтори через две минуты.")
            db.execute("INSERT INTO limits VALUES (?,?,1) ON CONFLICT(scope) DO UPDATE SET hits=hits+1",
                       (scope, now+120))


def configured_address(slot):
    value = os.getenv(SLOTS[slot], "")
    if not re.fullmatch(r"[A-Za-z0-9_+/-]{48}", value) or base64.urlsafe_b64decode(value)[0] not in (0x11, 0x51):
        raise finance.Error("Укажи адрес Mainnet с контрольной суммой.")
    return value, ton.address(value)


def _read(operation):
    try:
        return "found", operation()
    except ton.NotFound:
        return "not_found", None
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, BocError):
        return "unavailable", None


def _chain_account(owner):
    row = ton.get(f"blockchain/accounts/{owner}")
    if ton.address(row["address"]) != owner or row["status"] not in ("active", "uninit", "frozen", "nonexist"):
        raise ValueError
    nano_ton = ton.quantity(row["balance"], allow_zero=True)
    code_hash = ton.wallet_code_hash(Cell.one_from_boc(bytes.fromhex(row["code"]))) if row.get("code") else None
    return row["status"], format(Decimal(nano_ton)/10**9, "f"), code_hash


def _key(owner):
    key = ton.wallet_key(owner)
    if len(key) != 32:
        raise ValueError
    return key


def snapshot(owner):
    with ThreadPoolExecutor(max_workers=3) as pool:
        account = pool.submit(_read, lambda: _chain_account(owner))
        key = pool.submit(_read, lambda: _key(owner))
        jetton = pool.submit(_read, lambda: ton.jetton_wallet(owner))
        account_state, chain = account.result()
        key_state, live_key = key.result()
        jetton_state, usdt = jetton.result()
    status, ton_balance, code_hash = chain or (account_state, None, None)
    report = {"checked_at": finance.timestamp(), "account": status, "ton": ton_balance,
              "version": VERSIONS.get(code_hash, "Неизвестный контракт" if code_hash else None),
              "supported": code_hash in ton.WALLET_CODES if code_hash else None,
              "key": key_state, "jetton": jetton_state, "usdt": finance.display(usdt[1]) if usdt else None,
              "match": None, "network": None, "signature": "pending", "mainnet": "pending"}
    return report, live_key, code_hash


def _digest(payload):
    if not isinstance(payload, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32}", payload):
        raise finance.Error("Подключи кошелёк заново, чтобы получить подтверждение.")
    return hashlib.sha256(payload.encode()).hexdigest()


def action(admin, route, data, domain):
    if not allowed(admin):
        raise finance.Error("Диагностика недоступна.")
    now = finance.timestamp()
    limit(admin, route, now)
    if route == "state":
        cards = {}
        with transaction() as db:
            for slot in SLOTS:
                try:
                    address, owner = configured_address(slot)
                except finance.Error:
                    cards[slot] = {"checksum": False}
                    continue
                row = db.execute("SELECT report FROM results WHERE admin=? AND slot=? AND address=? AND domain=?",
                                 (admin, slot, owner, domain)).fetchone()
                cards[slot] = {"checksum": True, "address": address, "workchain": 0,
                               "report": json.loads(row[0]) if row else None}
        return {"cards": cards}

    slot = data.get("slot")
    if not isinstance(slot, str) or slot not in SLOTS:
        raise finance.Error("Выбери один из двух кошельков.")
    _, owner = configured_address(slot)
    if route == "challenge":
        report, _, _ = snapshot(owner)
        payload = secrets.token_urlsafe(24)
        with transaction() as db:
            issued = finance.timestamp()
            db.execute("DELETE FROM challenges WHERE issued<=?", (issued-ton.PROOF_TTL,))
            db.execute("INSERT OR REPLACE INTO challenges VALUES (?,?,?,?,?,?)",
                       (admin, slot, owner, domain, _digest(payload), issued))
        return {"payload": payload, "report": report}

    account, proof = data.get("account"), data.get("proof")
    if not isinstance(account, dict) or not isinstance(proof, dict):
        raise finance.Error("Кошелёк не передал подтверждение TON Connect.")
    digest = _digest(proof.get("payload"))
    with transaction() as db:
        row = db.execute("DELETE FROM challenges WHERE admin=? AND slot=? AND address=? AND domain=? "
                         "AND digest=? AND issued>? RETURNING issued",
                         (admin, slot, owner, domain, digest, now-ton.PROOF_TTL)).fetchone()
        if not row:
            raise finance.Error("Подтверждение истекло или уже использовано. Подключи кошелёк заново.")
    report, live_key, live_code = snapshot(owner)
    try:
        report["match"] = ton.address(account.get("address")) == owner
    except finance.Error:
        report["match"] = False
    report["network"] = account.get("chain") == "-239"
    if report["match"] and report["network"]:
        try:
            _, key, code = ton.proof_details(account, proof, domain, row[0], now)
        except ton.UnsupportedWallet:
            report["signature"] = "unsupported"
        except finance.Error:
            report["signature"] = "failed"
        else:
            report["signature"] = "confirmed"
            if not live_code:
                report["version"] = VERSIONS.get(code)
                report["supported"] = code in ton.WALLET_CODES
            if report["account"] == "active" and live_code == code and live_key == key:
                report["mainnet"] = "confirmed"
    else:
        report["signature"] = "failed"
    with transaction() as db:
        db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?)",
                   (admin, slot, owner, domain, json.dumps(report)))
    return {"report": report}
