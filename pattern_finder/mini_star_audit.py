"""Reconcile Stars invoice entries against locally confirmed payments."""
import argparse
import json
import threading
import time

import billing
import notifier
import partner_finance
import storage
import subscription_guard


FRESH_SECONDS = 300
_sync_lock = threading.Lock()


def require_fresh(db, now):
    row = db.execute("SELECT state,successful FROM fc_stars_sync WHERE id=1").fetchone()
    if not row or row[0] != "ok" or row[1] is None or not 0 <= now-row[1] <= FRESH_SECONDS:
        raise partner_finance.Error("Нужна свежая успешная сверка Stars. Выполни её перед переводом.")


def refresh(api=None, now=None):
    if not _sync_lock.acquire(blocking=False):
        raise partner_finance.Error("Сверка Stars уже выполняется. Повтори после завершения.")
    try:
        with storage.transaction() as db:
            partner_finance.schema(db)
            db.execute("INSERT INTO fc_stars_sync VALUES (1,'running',?,NULL) "
                       "ON CONFLICT(id) DO UPDATE SET state='running',started=excluded.started",
                       (partner_finance.timestamp(now),))
        try:
            with storage.transaction() as db:
                max_pages = partner_finance.number(db, "REFERRAL_STARS_MAX_PAGES", 40, 1, 200)
            def page(offset):
                result = (api or notifier.api_post)("getStarTransactions", {"offset": offset, "limit": 100})
                rows = result["result"]["transactions"]
                if result.get("ok") is not True or not isinstance(rows, list) or len(rows) > 100:
                    raise ValueError
                return rows

            deadline = time.monotonic()+45
            first = page(0)
            rows = list(first)
            last = first
            for index in range(1, max_pages+1):
                if len(last) < 100:
                    break
                if time.monotonic() > deadline:
                    raise ValueError
                last = page(100*index)
                rows.extend(last)
            if len(last) == 100 or (len(first) == 100 and page(0) != first) or time.monotonic() > deadline:
                raise ValueError
            with storage.transaction() as db:
                partner_finance.replay_payments(db, billing.admins())
                written = _reconcile(db, rows)
                db.execute("UPDATE fc_stars_sync SET state='ok',successful=? WHERE id=1", (partner_finance.timestamp(now),))
            return written
        except Exception:
            with storage.transaction() as db:
                db.execute("UPDATE fc_stars_sync SET state='failed' WHERE id=1")
            raise partner_finance.Error("Журнал Stars не удалось полностью сверить. Переводы приостановлены.") from None
    finally:
        _sync_lock.release()


def reconcile(transactions):
    with storage.transaction() as db:
        partner_finance.schema(db)
        return _reconcile(db, transactions)


def _reconcile(db, transactions):
    written = 0
    db.execute("""CREATE TABLE IF NOT EXISTS mini_star_audit (
        charge TEXT NOT NULL REFERENCES payments(id), event TEXT NOT NULL, at INTEGER NOT NULL,
        ledger_nanos INTEGER NOT NULL, commission_nanos INTEGER, rate INTEGER,
        PRIMARY KEY(charge,event))""")
    db.execute("""CREATE TABLE IF NOT EXISTS fc_fragment_withdrawals (
        id TEXT PRIMARY KEY, stars INTEGER NOT NULL, confirmed INTEGER NOT NULL)""")
    for tx in transactions:
        source, receiver = tx.get("source"), tx.get("receiver")
        if bool(source) == bool(receiver):
            continue
        withdrawal = (receiver or {}).get("withdrawal_state") or {}
        if receiver and receiver.get("type") == "fragment" and withdrawal.get("type") == "succeeded":
            stars = abs(tx["amount"])
            if stars and not tx.get("nanostar_amount"):
                db.execute("INSERT OR IGNORE INTO fc_fragment_withdrawals VALUES (?,?,?)",
                           (tx["id"], stars, withdrawal["date"]))
            continue
        peer = source or receiver
        if peer.get("type") != "user" or peer.get("transaction_type") != "invoice_payment":
            continue
        charge = tx.get("id")
        row = db.execute("SELECT chat_id,data FROM payments WHERE id=? AND source='stars' AND currency='XTR'",
                         (charge,)).fetchone()
        if not row or str(peer.get("user", {}).get("id")) != row[0]:
            continue
        if peer.get("invoice_payload") and peer["invoice_payload"] != json.loads(row[1]).get("invoice_payload"):
            continue
        affiliate = peer.get("affiliate")
        delta = tx["amount"] * 10**9 + tx.get("nanostar_amount", 0)
        commission = affiliate["amount"] * 10**9 + affiliate.get("nanostar_amount", 0) if affiliate else None
        event = "payment" if source else "refund"
        cursor = db.execute("INSERT OR IGNORE INTO mini_star_audit VALUES (?,?,?,?,?,?)",
                            (charge, event, tx["date"], abs(delta) if source else -abs(delta), commission,
                             affiliate["commission_per_mille"] if affiliate else None))
        written += cursor.rowcount
        if event == "payment" and commission:
            db.execute("UPDATE mini_star_audit SET commission_nanos=?,rate=? WHERE charge=? AND event='payment' "
                       "AND COALESCE(commission_nanos,0)=0", (commission, affiliate['commission_per_mille'], charge))
        if event == "refund":
            partner_finance.reverse(db, charge, tx["date"], billing.admins())
    return written


def summary():
    with storage.transaction() as db:
        return db.execute("""SELECT a.event, COUNT(*), SUM(p.amount), SUM(a.ledger_nanos),
                             COUNT(a.commission_nanos), SUM(a.commission_nanos)
                             FROM mini_star_audit a JOIN payments p ON p.id=a.charge GROUP BY a.event""").fetchall()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сверка подтверждённых платежей и возвратов Stars")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fragment", action="store_true", help="Показать подтверждённые выводы Fragment")
    parser.add_argument("--charge", help="Показать конкретный Telegram charge ID")
    parser.add_argument("--conflicts", action="store_true", help="Показать платежи для ручной сверки")
    args = parser.parse_args()
    if args.conflicts:
        with storage.transaction() as db:
            subscription_guard.schema(db)
            for row in db.execute("SELECT charge,buyer,recurring,state FROM pro_conflicts WHERE state!='refunded'"):
                print(*row)
        raise SystemExit(0)
    if args.offset < 0 or not notifier.TOKEN:
        raise SystemExit("Некорректный offset или отсутствует TELEGRAM_TOKEN")
    try:
        reply = notifier.api_post("getStarTransactions", {"offset": args.offset, "limit": 100})
    except Exception:
        raise SystemExit("Не удалось прочитать журнал Telegram") from None
    if not reply.get("ok"):
        raise SystemExit("Telegram отклонил запрос журнала")
    print("Новых записей:", reconcile(reply["result"]["transactions"]))
    if args.fragment:
        with storage.transaction() as db:
            for row in db.execute("SELECT id,stars,confirmed FROM fc_fragment_withdrawals ORDER BY confirmed DESC LIMIT 30"):
                print(*row)
        raise SystemExit(0)
    if args.charge:
        with storage.transaction() as db:
            rows = db.execute("SELECT a.event,p.amount,a.ledger_nanos,a.commission_nanos,p.status "
                              "FROM mini_star_audit a JOIN payments p ON p.id=a.charge WHERE a.charge=?",
                              (args.charge,)).fetchall()
        for event, gross, ledger, fee, status in rows:
            print(event, "покупка:", gross, "движение Telegram:", ledger / 10**9,
                  "комиссия:", fee / 10**9 if fee is not None else "неизвестно", "статус:", status)
        if not rows:
            print("Операция пока не сверена. Проверь offset и наличие локальной оплаты.")
        raise SystemExit(0)
    for event, count, gross, delta, known, fee in summary():
        print(event, "записей:", count, "сумма исходных оплат:", gross,
              "движение по журналу:", delta / 10**9, "комиссия подтверждена для:", known,
              "сумма подтверждённой комиссии:", fee / 10**9 if fee is not None else "неизвестно")
