"""Reconcile Stars invoice entries against locally confirmed payments."""
import argparse
import json

import notifier
import storage


def reconcile(transactions):
    written = 0
    with storage.transaction() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS mini_star_audit (
            charge TEXT NOT NULL REFERENCES payments(id), event TEXT NOT NULL, at INTEGER NOT NULL,
            ledger_nanos INTEGER NOT NULL, commission_nanos INTEGER, rate INTEGER,
            PRIMARY KEY(charge,event))""")
        for tx in transactions:
            source, receiver = tx.get("source"), tx.get("receiver")
            if bool(source) == bool(receiver):
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
    return written


def summary():
    with storage.transaction() as db:
        return db.execute("""SELECT a.event, COUNT(*), SUM(p.amount), SUM(a.ledger_nanos),
                             COUNT(a.commission_nanos), SUM(a.commission_nanos)
                             FROM mini_star_audit a JOIN payments p ON p.id=a.charge GROUP BY a.event""").fetchall()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сверка подтверждённых платежей и возвратов Stars")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.offset < 0 or not notifier.TOKEN:
        raise SystemExit("Некорректный offset или отсутствует TELEGRAM_TOKEN")
    try:
        reply = notifier.api_post("getStarTransactions", {"offset": args.offset, "limit": 100})
    except Exception:
        raise SystemExit("Не удалось прочитать журнал Telegram") from None
    if not reply.get("ok"):
        raise SystemExit("Telegram отклонил запрос журнала")
    print("Новых записей:", reconcile(reply["result"]["transactions"]))
    for event, count, gross, delta, known, fee in summary():
        print(event, "записей:", count, "сумма исходных оплат:", gross,
              "движение по журналу:", delta / 10**9, "комиссия подтверждена для:", known,
              "сумма подтверждённой комиссии:", fee / 10**9 if fee is not None else "неизвестно")
