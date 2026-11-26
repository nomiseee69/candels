"""Reconcile Stars invoice entries against locally confirmed payments."""
import argparse
import json

import notifier
import partner_finance
import storage
import subscription_guard


def reconcile(transactions):
    written = 0
    with storage.transaction() as db:
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
            if event == "refund":
                partner_finance.reverse(db, charge, tx["date"])
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
