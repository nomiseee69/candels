"""Private /admin operations for the partners ledger."""
import re

import partner_finance as f
import partner_payouts as payouts
import partner_ton as ton
import storage


def report(partner=None):
    with storage.transaction() as db:
        f.schema(db)
        if partner is not None:
            wallet = db.execute("SELECT address,verified FROM fc_wallets WHERE partner=?", (partner,)).fetchone()
            invites = db.execute("SELECT COUNT(*) FROM fc_referral_attributions WHERE partner=?", (partner,)).fetchone()[0]
            earnings = db.execute("SELECT c.charge,c.plan,c.stars,c.mature,a.amount,r.at FROM fc_referral_commissions c "
                                  "LEFT JOIN fc_allocations a USING(charge) LEFT JOIN fc_reversals r USING(charge) "
                                  "WHERE c.partner=? ORDER BY c.at DESC LIMIT 8", (partner,)).fetchall()
            risks = db.execute("SELECT flag FROM fc_risk WHERE partner=? AND resolved IS NULL", (partner,)).fetchall()
            rows = db.execute("SELECT id,partner,amount,status,address,tx FROM fc_payout_requests WHERE partner=? "
                              "ORDER BY id DESC LIMIT 8", (partner,)).fetchall()
            lines = [f"Партнёр {partner} · приглашений {invites}", f"Кошелёк: {wallet[0] if wallet and wallet[1] else '—'}",
                     "Риски: " + (", ".join(r[0] for r in risks) or "—")]
            lines += [f"{c} · {p} · {s} Stars · " + ("возврат" if r else f"{f.display(a)} USDT" if a is not None
                       else "ожидается обеспечение" if mature <= f.timestamp() else "ожидание")
                      for c, p, s, mature, a, r in earnings]
        else:
            counts = [db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in (
                "fc_partners", "fc_referral_attributions", "fc_referral_commissions")]
            funded = db.execute("SELECT COALESCE(SUM(amount),0) FROM fc_treasury_ledger").fetchone()[0]
            stats = db.execute("SELECT COUNT(*),COALESCE(SUM(usdt),0),COALESCE(SUM(stars),0),"
                               "COALESCE(SUM(ton_nano),0),COALESCE(SUM(costs),0) FROM fc_stars_batches").fetchone()
            pending = db.execute("SELECT COUNT(*),COUNT(estimate),COALESCE(SUM(estimate),0) FROM fc_referral_commissions c WHERE "
                                 "NOT EXISTS (SELECT 1 FROM fc_allocations WHERE charge=c.charge) AND "
                                 "NOT EXISTS (SELECT 1 FROM fc_reversals WHERE charge=c.charge)").fetchone()
            commissions = db.execute("SELECT COALESCE(SUM(amount),0) FROM fc_allocations").fetchone()[0]
            risks = db.execute("SELECT COUNT(*) FROM fc_risk WHERE resolved IS NULL").fetchone()[0]
            rows = db.execute("SELECT id,partner,amount,status,address,tx FROM fc_payout_requests "
                              "ORDER BY status IN ('requested','approved','submitted') DESC,id DESC LIMIT 8").fetchall()
            lines = [f"🤝 Finder Candles Partners · партнёры {counts[0]} · приглашения {counts[1]} · покупки {counts[2]}",
                     f"Ожидающие начисления: {pending[0]} · отметки риска: {risks}",
                     f"Оценка ожидающих: {f.display(pending[2])} USDT для {pending[1]} из {pending[0]}"
                     if pending[1] else "Оценка ожидающих начислений пока недоступна.",
                     f"Всего обеспечено комиссий до учёта возвратов: {f.display(commissions)} USDT",
                     f"Подтверждённые пополнения: {f.display(funded)} USDT",
                     f"Сверки: {stats[0]} · выведено {stats[2]} Stars · получено {stats[3]} наноTON · "
                     f"пополнено {f.display(stats[1])} USDT · вне пополнений учтено расходов {f.display(stats[4])} USDT"]
        available, reserved, paid, debt = f.balances(db, partner)
        lines.append(f"Доступно по журналу {f.display(available)} · резерв {f.display(reserved)} · "
                     f"выплачено {f.display(paid)} · корректировки {f.display(debt)} USDT")
    for key, user, units, status, destination, tx in rows:
        lines.append(f"№{key} · {user} · {f.display(units)} USDT · {status}\n{destination}" + (f"\n{tx}" if tx else ""))
    return "\n".join(lines)[:3900]


def _notes(text):
    if not text.strip():
        raise f.Error("Укажи причину операции.")
    return text.strip()[:200]


def _id(text):
    if not re.fullmatch(r"[1-9][0-9]{0,15}", text):
        raise f.Error("Укажи числовой ID.")
    return int(text)


def _manage(actor, partner, action, reason):
    with storage.transaction() as db:
        f.schema(db)
        payouts._partner(db, partner)
        if action == "clear-risk":
            db.execute("UPDATE fc_risk SET resolved=? WHERE partner=? AND resolved IS NULL", (f.timestamp(), partner))
        else:
            db.execute("UPDATE fc_partners SET blocked=? WHERE id=?", (action == "block", partner))
        f.audit(db, actor, partner, action, reason, f.timestamp())


def treasury_report():
    owner = ton.treasury()
    _, live = ton.jetton_wallet(owner)
    with storage.transaction() as db:
        f.schema(db)
        available, reserved, _, _ = f.balances(db)
    return (f"Резерв TON Mainnet {owner}\nПроверенный остаток {f.display(live)} USDT\n"
            f"Обязательства {f.display(available+reserved)} USDT\n"
            f"Свободные средства по данным TON {f.display(max(0,live-available-reserved))} USDT")


def reconciliation_report():
    with storage.transaction() as db:
        f.schema(db)
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        stars = dict(db.execute("SELECT event,SUM(ledger_nanos) FROM mini_star_audit GROUP BY event")) \
            if "mini_star_audit" in tables else {}
        withdrawn = db.execute("SELECT COALESCE(SUM(stars),0) FROM fc_fragment_withdrawals").fetchone()[0] \
            if "fc_fragment_withdrawals" in tables else 0
        batches = db.execute("SELECT id,at,source_ref,stars,ton_nano,usdt,costs,tx FROM fc_stars_batches "
                             "ORDER BY id DESC LIMIT 8").fetchall()
    lines = ["📊 Сверка Stars", f"Импортированные покупки: {stars.get('payment', 0)} наноStars · "
             f"возвраты: {abs(stars.get('refund', 0))} наноStars",
             f"Подтверждено вывода через Fragment: {withdrawn} Stars",
             "Доступную к выводу сумму проверь в Telegram перед сверкой."]
    for key, at, ref, count, nano, usdt, costs, tx in batches:
        lines.append(f"№{key} · {at} UTC Unix · Fragment {ref} · {count} Stars · {nano} наноTON · "
                     f"{f.display(usdt)} USDT · расходы {f.display(costs)} USDT\n{tx}")
    return "\n".join(lines)[:3900]


def payout_report(key):
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT partner,amount,address,at,due,status,tx FROM fc_payout_requests WHERE id=?",
                         (key,)).fetchone()
        if not row:
            raise f.Error("Заявка не найдена.")
        actions = db.execute("SELECT at,event,reason FROM fc_admin_audit WHERE subject=? "
                             "AND event IN ('requested','approved','rejected','cancelled','dispatch','tx_submitted','paid') "
                             "ORDER BY id", (str(key),)).fetchall()
    partner, units, address, at, due, status, tx = row
    lines = [f"Заявка №{key} · партнёр {partner} · {f.display(units)} USDT · {status}",
             f"Создана {at} · срок {due} (UTC Unix)", address, tx or "TXID пока не зарегистрирован."]
    return "\n".join(lines + [f"{stamp} · {event} · {reason}" for stamp, event, reason in actions])[:3900]


def command(actor, argument):
    action, _, args = argument.strip().partition(" ")
    if not action:
        return report()
    if action == "help":
        return ("🤝 /partners\n/partners user ID\n/partners payout №\n/partners block|unblock|clear-risk ID причина\n"
                "/partners approve|reject|dispatch № причина\n/partners tx № TXID\n/partners verify №\n"
                "/partners treasury\n/partners reconciliation\n/partners topup TXID причина\n"
                "/partners batch TXID FRAGMENT_REF STARS TON_NANO COST_USDT CHARGE[,CHARGE]\n"
                "/partners rate MICRO_USDT_PER_STAR источник\n/partners set ПАРАМЕТР ЗНАЧЕНИЕ причина\n"
                "/partners audit [после_ID]")
    if action == "user":
        return report(_id(args))
    if action == "payout":
        return payout_report(_id(args))
    if action == "reconciliation":
        return reconciliation_report()
    if action == "treasury":
        return treasury_report()
    if action == "audit":
        start = _id(args) if args else 0
        with storage.transaction() as db:
            f.schema(db)
            rows = db.execute("SELECT id,at,actor,subject,event,reason FROM fc_admin_audit WHERE id>? "
                              "ORDER BY id LIMIT 20", (start,)).fetchall()
        return "\n".join(" · ".join(map(str, row)) for row in rows)[:3900] or "Записей дальше нет."
    if action in ("block", "unblock", "clear-risk", "approve", "reject", "dispatch"):
        key, _, reason = args.partition(" ")
        key, reason = _id(key), _notes(reason)
        if action in ("block", "unblock", "clear-risk"):
            _manage(actor, key, action, reason)
        elif action == "dispatch":
            payouts.dispatch(actor, key, reason)
            return "Резерв заморожен. Сверь реквизиты в /partners и переведи вручную. Затем укажи TXID."
        else:
            payouts.decision(actor, key, action == "approve", reason)
        return "✅ Решение записано в аудит."
    if action == "tx":
        key, _, txid = args.partition(" ")
        payouts.submit(actor, _id(key), txid)
        return "TXID записан. Для проверки выполни /partners verify НОМЕР."
    if action == "verify":
        payouts.verify_payout(actor, _id(args))
        return "✅ Перевод подтверждён в TON."
    if action == "topup":
        txid, _, reason = args.partition(" ")
        return f"✅ Подтверждено пополнение {f.display(payouts.record_topup(actor, txid, _notes(reason)))} USDT."
    if action == "batch":
        parts = args.split()
        if len(parts) != 6 or not all(re.fullmatch(r"[0-9]{1,17}", n) for n in parts[2:4]):
            raise f.Error("Формат сверки указан в /partners help.")
        txid, ref, stars, nano, costs, charges = parts
        return f"✅ Сверка №{payouts.settle_batch(actor, txid, ref, int(stars), int(nano), f.amount(costs), charges.split(','))}."
    if action == "rate":
        value, _, evidence = args.partition(" ")
        f.set_rate(actor, _id(value), _notes(evidence))
        return "✅ Справочный курс сохранён."
    if action == "set":
        key, _, tail = args.partition(" ")
        value, _, reason = tail.partition(" ")
        f.set_setting(actor, key, value, _notes(reason))
        return "✅ Настройка сохранена."
    raise f.Error("Список команд доступен в /partners help.")
