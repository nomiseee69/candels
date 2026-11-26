"""Finder Candles Partners. USDT amounts are integer micro-USDT."""
import json
import os
import re
import secrets
import time
from decimal import Decimal

import storage

DAY = 86400
UNIT = 1_000_000
ACTIVE = ("requested", "approved", "submitted")
SCHEMA = """
CREATE TABLE IF NOT EXISTS fc_partners (id INTEGER PRIMARY KEY, blocked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS fc_referral_codes (partner INTEGER PRIMARY KEY REFERENCES fc_partners(id), code TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS fc_referral_attributions (buyer INTEGER PRIMARY KEY, partner INTEGER NOT NULL REFERENCES fc_partners(id),
 at INTEGER NOT NULL, source TEXT NOT NULL, CHECK(buyer != partner));
CREATE INDEX IF NOT EXISTS fc_attribution_partner ON fc_referral_attributions(partner,at);
CREATE TABLE IF NOT EXISTS fc_referral_commissions (charge TEXT PRIMARY KEY REFERENCES payments(id), buyer INTEGER UNIQUE NOT NULL,
 partner INTEGER NOT NULL REFERENCES fc_partners(id), stars INTEGER NOT NULL, plan TEXT NOT NULL, discount INTEGER,
 at INTEGER NOT NULL, mature INTEGER NOT NULL, rate_bps INTEGER NOT NULL, quote INTEGER, estimate INTEGER);
CREATE TABLE IF NOT EXISTS fc_referral_ledger (event TEXT NOT NULL, ref TEXT NOT NULL,
 partner INTEGER NOT NULL REFERENCES fc_partners(id), at INTEGER NOT NULL, available INTEGER NOT NULL DEFAULT 0,
 reserved INTEGER NOT NULL DEFAULT 0, paid INTEGER NOT NULL DEFAULT 0, debt INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(event,ref));
CREATE INDEX IF NOT EXISTS fc_ledger_partner ON fc_referral_ledger(partner);
CREATE TABLE IF NOT EXISTS fc_wallets (partner INTEGER PRIMARY KEY REFERENCES fc_partners(id), address TEXT NOT NULL,
 connected INTEGER NOT NULL, unlock INTEGER NOT NULL, verified INTEGER NOT NULL, network TEXT NOT NULL CHECK(network='TON_MAINNET'));
CREATE INDEX IF NOT EXISTS fc_wallet_address ON fc_wallets(address);
CREATE TABLE IF NOT EXISTS fc_ton_challenges (partner INTEGER PRIMARY KEY REFERENCES fc_partners(id), token TEXT UNIQUE NOT NULL,
 issued INTEGER NOT NULL, expires INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fc_payout_requests (id INTEGER PRIMARY KEY, partner INTEGER NOT NULL REFERENCES fc_partners(id),
 amount INTEGER NOT NULL CHECK(amount>0), address TEXT NOT NULL, source TEXT NOT NULL, at INTEGER NOT NULL, due INTEGER NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('requested','approved','submitted','paid','rejected','cancelled')),
 reason TEXT, tx TEXT UNIQUE, submitted INTEGER);
CREATE UNIQUE INDEX IF NOT EXISTS fc_open_payout ON fc_payout_requests(partner)
 WHERE status IN ('requested','approved','submitted');
CREATE TABLE IF NOT EXISTS fc_payout_transactions (tx TEXT PRIMARY KEY, request INTEGER UNIQUE NOT NULL REFERENCES fc_payout_requests(id),
 destination_tx TEXT UNIQUE NOT NULL, at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fc_treasury_ledger (tx TEXT PRIMARY KEY, destination_tx TEXT UNIQUE NOT NULL, amount INTEGER NOT NULL,
 at INTEGER NOT NULL, source TEXT NOT NULL, destination TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fc_stars_batches (id INTEGER PRIMARY KEY, tx TEXT UNIQUE NOT NULL REFERENCES fc_treasury_ledger(tx),
 source_ref TEXT UNIQUE NOT NULL, stars INTEGER NOT NULL, ton_nano INTEGER NOT NULL, usdt INTEGER NOT NULL, costs INTEGER NOT NULL,
 at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fc_allocations (charge TEXT PRIMARY KEY REFERENCES fc_referral_commissions(charge),
 batch INTEGER NOT NULL REFERENCES fc_stars_batches(id), amount INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fc_reversals (charge TEXT PRIMARY KEY REFERENCES fc_referral_commissions(charge), at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fc_admin_audit (id INTEGER PRIMARY KEY, at INTEGER NOT NULL, actor INTEGER,
 subject TEXT NOT NULL, event TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fc_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fc_rate_history (id INTEGER PRIMARY KEY, at INTEGER NOT NULL, actor INTEGER NOT NULL,
 rate INTEGER NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fc_risk (partner INTEGER NOT NULL REFERENCES fc_partners(id), flag TEXT NOT NULL,
 at INTEGER NOT NULL, resolved INTEGER, PRIMARY KEY(partner,flag));
CREATE TABLE IF NOT EXISTS fc_notifications (id INTEGER PRIMARY KEY, chat INTEGER NOT NULL, text TEXT NOT NULL,
 attempted INTEGER, sent INTEGER);
"""


class Error(ValueError):
    pass


def schema(db):
    for statement in SCHEMA.strip().rstrip(";").split(";"):
        db.execute(statement)
    for table in ("fc_referral_ledger", "fc_treasury_ledger", "fc_admin_audit", "fc_allocations", "fc_stars_batches",
                  "fc_payout_transactions", "fc_rate_history", "fc_reversals"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} ON {table} "
                       "BEGIN SELECT RAISE(ABORT,'append-only'); END")


def timestamp(now=None):
    return int(time.time() if now is None else now)


def flag(name):
    return os.getenv(name, "0") == "1"


def cutover():
    raw = os.getenv("REFERRAL_NATIVE_ENDED_AT", "")
    return int(raw) if raw.isdecimal() and len(raw) <= 12 else 0


def enabled(now=None):
    # Existing native referrals may remain eligible after the program is ended.
    return (flag("REFERRALS_ENABLED") and not flag("MINI_APP_AFFILIATE_ACTIVE")
            and 0 < cutover() <= timestamp(now) - 32 * DAY)


def get_setting(db, key, default):
    row = db.execute("SELECT value FROM fc_settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else os.getenv(key, str(default))


def number(db, key, default, low, high):
    value = get_setting(db, key, default)
    if not value.isdecimal() or len(value) > 10 or not low <= int(value) <= high:
        raise Error("Проверь настройки партнёрской программы.")
    return int(value)


def amount(text):
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,6})?", text):
        raise Error("Укажи сумму с точностью до шести знаков после точки.")
    return int(Decimal(text) * UNIT)


def display(value):
    return format(Decimal(value) / UNIT, "f")


def audit(db, actor, subject, event, reason, at):
    db.execute("INSERT INTO fc_admin_audit(at,actor,subject,event,reason) VALUES (?,?,?,?,?)",
               (at, actor, str(subject), event, reason))


def notify(db, chat, text):
    db.execute("INSERT INTO fc_notifications(chat,text) VALUES (?,?)", (chat, text))


def set_setting(actor, key, value, reason):
    limits = {
        "REFERRAL_COMMISSION_PERCENT": (1, 100), "MINIMUM_PAYOUT_USDT": (10, 1000000),
        "REFERRAL_HOLD_DAYS": (21, 365), "WALLET_CHANGE_COOLDOWN_HOURS": (48, 720),
        "REFERRAL_PAYOUT_INTERVAL_DAYS": (1, 30), "REFERRAL_REGISTRATION_OPEN": (0, 1),
        "REFERRAL_WITHDRAWALS_OPEN": (0, 1),
    }
    if key not in limits or not value.isdecimal() or len(value) > 10 or not limits[key][0] <= int(value) <= limits[key][1]:
        raise Error("Недопустимая настройка.")
    with storage.transaction() as db:
        schema(db)
        db.execute("INSERT OR REPLACE INTO fc_settings VALUES (?,?)", (key, value))
        audit(db, actor, key, "setting", f"{value} · {reason}", timestamp())


def set_rate(actor, rate, evidence):
    if not 0 < rate <= UNIT or not evidence:
        raise Error("Нужен курс в микро-USDT за Stars и источник.")
    with storage.transaction() as db:
        schema(db)
        db.execute("INSERT INTO fc_rate_history(at,actor,rate,source) VALUES (?,?,?,?)",
                   (timestamp(), actor, rate, evidence))
        audit(db, actor, "Stars", "rate", evidence, timestamp())


def link(partner):
    with storage.transaction() as db:
        schema(db)
        row = db.execute("SELECT code FROM fc_referral_codes WHERE partner=?", (partner,)).fetchone()
        if row:
            return "https://t.me/findercandles_bot?start=fc_" + row[0]
        if not enabled() or get_setting(db, "REFERRAL_REGISTRATION_OPEN", 1) != "1":
            raise Error("Регистрация партнёров пока закрыта.")
        db.execute("INSERT OR IGNORE INTO fc_partners(id) VALUES (?)", (partner,))
        code = secrets.token_urlsafe(24)
        db.execute("INSERT INTO fc_referral_codes VALUES (?,?)", (partner, code))
        return "https://t.me/findercandles_bot?start=fc_" + code


def risk(db, partner, flag_name, at):
    db.execute("INSERT INTO fc_risk VALUES (?,?,?,NULL) ON CONFLICT(partner,flag) "
               "DO UPDATE SET at=excluded.at,resolved=NULL WHERE fc_risk.resolved IS NOT NULL", (partner, flag_name, at))


def attribute(buyer, code, now=None):
    at = timestamp(now)
    if not enabled(at) or buyer <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]{32}", code):
        return False
    with storage.transaction() as db:
        schema(db)
        row = db.execute("SELECT id,blocked FROM fc_referral_codes JOIN fc_partners ON id=partner WHERE code=?",
                         (code,)).fetchone()
        if not row or row[0] == buyer or row[1] or get_setting(db, "REFERRAL_REGISTRATION_OPEN", 1) != "1":
            return False
        if any(db.execute(sql, (buyer,)).fetchone() for sql in (
                "SELECT 1 FROM users WHERE chat_id=CAST(? AS TEXT)", "SELECT 1 FROM payments WHERE chat_id=CAST(? AS TEXT)",
                "SELECT 1 FROM fc_partners WHERE id=?", "SELECT 1 FROM fc_referral_attributions WHERE buyer=?")):
            return False
        db.execute("INSERT INTO fc_referral_attributions VALUES (?,?,?,'telegram_start')", (buyer, row[0], at))
        storage.update_user_in(db, buyer, lambda user: None)
        if db.execute("SELECT COUNT(*) FROM fc_referral_attributions WHERE partner=? AND at>=?",
                      (row[0], at - 3600)).fetchone()[0] >= 10:
            risk(db, row[0], "rapid_invites", at)
        notify(db, row[0], "👥 По твоей ссылке зарегистрировался пользователь.")
        return True


def record_purchase(db, buyer, charge, plan, payment, now, discount=None):
    at = timestamp(now)
    if (not enabled(at) or payment.get("currency") != "XTR" or payment.get("test") or payment.get("is_test")
            or flag("REFERRAL_TEST_MODE") or type(payment.get("total_amount")) is not int
            or not 0 < payment["total_amount"] <= 10**8 or plan not in ("month", "year")
            or (payment.get("is_recurring") and not payment.get("is_first_recurring"))):
        return
    schema(db)
    row = db.execute("SELECT partner FROM fc_referral_attributions WHERE buyer=? AND at<=?", (buyer, at)).fetchone()
    if not row:
        return
    # A prior settled payment, even a later-refunded one, consumes the first purchase.
    if db.execute("SELECT 1 FROM payments WHERE chat_id=? AND id!=? AND source='stars' AND currency='XTR' "
                  "AND plan IN ('month','year') AND status IN ('succeeded','refunded') LIMIT 1",
                  (str(buyer), charge)).fetchone():
        return
    try:
        bps = 100 * number(db, "REFERRAL_COMMISSION_PERCENT", 10, 1, 100)
        hold = number(db, "REFERRAL_HOLD_DAYS", 21, 21, 365)
    except Error:
        audit(db, None, charge, "configuration", "Комиссия требует сверки настроек", at)
        return
    quote = db.execute("SELECT id,rate FROM fc_rate_history WHERE at BETWEEN ? AND ? ORDER BY id DESC LIMIT 1",
                       (at-30*DAY, at)).fetchone()
    stars = payment["total_amount"]
    estimate = stars * quote[1] * bps // 10000 if quote else None
    result = db.execute("INSERT OR IGNORE INTO fc_referral_commissions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (charge, buyer, row[0], stars, plan, discount, at, at+hold*DAY, bps,
                         quote[0] if quote else None, estimate))
    if result.rowcount:
        notify(db, row[0], "⭐ Первая покупка подтверждена. Начисление ожидает сверки и обеспечения.")
        if db.execute("SELECT COUNT(*) FROM fc_referral_commissions WHERE partner=? AND stars=? AND at>=?",
                      (row[0], stars, at - 3600)).fetchone()[0] >= 8:
            risk(db, row[0], "similar_purchases", at)


def balances(db, partner=None):
    sql = "SELECT " + ",".join(f"COALESCE(SUM({key}),0)" for key in ("available","reserved","paid","debt"))
    return db.execute(sql + " FROM fc_referral_ledger" + (" WHERE partner=?" if partner is not None else ""),
                      (partner,) if partner is not None else ()).fetchone()


def journal(db, event, ref, partner, at, *, available=0, reserved=0, paid=0, debt=0):
    db.execute("INSERT INTO fc_referral_ledger VALUES (?,?,?,?,?,?,?,?)",
               (event, str(ref), partner, at, available, reserved, paid, debt))


def credit(db, event, ref, partner, at, value, reserved=0):
    debt = balances(db, partner)[3]
    offset = min(debt, value)
    journal(db, event, ref, partner, at, available=value-offset, reserved=reserved, debt=-offset)


def reverse(db, charge, now=None):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='fc_referral_commissions'").fetchone():
        return
    at = timestamp(now)
    row = db.execute("SELECT partner FROM fc_referral_commissions WHERE charge=?", (charge,)).fetchone()
    if not row or not db.execute("INSERT OR IGNORE INTO fc_reversals VALUES (?,?)", (charge, at)).rowcount:
        return
    partner, = row
    payout = db.execute("SELECT id,amount FROM fc_payout_requests WHERE partner=? AND status IN ('requested','approved')",
                        (partner,)).fetchone()
    if payout:
        db.execute("UPDATE fc_payout_requests SET status='cancelled',reason='Возврат покупки' WHERE id=?", (payout[0],))
        credit(db, "cancel", payout[0], partner, at, payout[1], reserved=-payout[1])
        audit(db, None, payout[0], "cancelled", "Возврат покупки", at)
    funded = db.execute("SELECT amount FROM fc_allocations WHERE charge=?", (charge,)).fetchone()
    if funded:
        debit = min(funded[0], balances(db, partner)[0])
        journal(db, "reversal", charge, partner, at, available=-debit, debt=funded[0]-debit)
    if db.execute("SELECT COUNT(*) FROM fc_reversals r JOIN fc_referral_commissions c USING(charge) "
                  "WHERE c.partner=? AND r.at>=?", (partner, at-7*DAY)).fetchone()[0] >= 3:
        risk(db, partner, "repeated_refunds", at)
    audit(db, None, charge, "refund", "Возврат Stars", at)
    notify(db, partner, "↩️ Начисление скорректировано после возврата покупки.")



def send_notifications(api):
    with storage.transaction() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='fc_notifications'").fetchone():
            return
        now = timestamp()
        rows = db.execute("SELECT id,chat,text FROM fc_notifications WHERE sent IS NULL "
                          "AND (attempted IS NULL OR attempted<?) ORDER BY id LIMIT 20", (now-3600,)).fetchall()
        db.executemany("UPDATE fc_notifications SET attempted=? WHERE id=?", ((now, row[0]) for row in rows))
    for key, chat, text in rows:
        reply = api("sendMessage", {"chat_id": chat, "text": text})
        if reply.get("ok") is True or reply.get("error_code") in (400, 403):
            with storage.transaction() as db:
                db.execute("UPDATE fc_notifications SET sent=? WHERE id=?", (timestamp(), key))
