"""Private /admin operations for the partners ledger."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import mini_star_audit as stars
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
        sync = db.execute("SELECT state,successful FROM fc_stars_sync WHERE id=1").fetchone()
    lines = ["📊 Сверка Stars", f"Импортированные покупки: {stars.get('payment', 0)} наноStars · "
             f"возвраты: {abs(stars.get('refund', 0))} наноStars",
             f"Подтверждено вывода через Fragment: {withdrawn} Stars",
             "Доступную к выводу сумму проверь в Telegram перед сверкой."]
    if sync:
        lines.append(f"Сверка · {sync[0]} · последняя успешная {_date(sync[1])}")
    for key, at, ref, count, nano, usdt, costs, tx in batches:
        lines.append(f"№{key} · {at} UTC Unix · Fragment {ref} · {count} Stars · {nano} наноTON · "
                     f"{f.display(usdt)} USDT · расходы {f.display(costs)} USDT\n{tx}")
    return "\n".join(lines)[:3900]


def _date(at):
    return datetime.fromtimestamp(at, timezone.utc).strftime("%d.%m.%Y %H:%M UTC") if at else "—"


def _payout_info(key):
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT partner,amount,address,at,due,status,tx FROM fc_payout_requests WHERE id=?",
                         (key,)).fetchone()
        if not row:
            raise f.Error("Заявка не найдена.")
        actions = db.execute("SELECT at,event,reason FROM fc_admin_audit WHERE subject=? "
                             "AND event IN ('requested','approved','rejected','cancelled','dispatch','release','reset_preparation',"
                             "'tx_submitted','tx_replacement_proposed','paid','refund_after_dispatch','verify_failed') "
                             "ORDER BY id DESC LIMIT 10", (str(key),)).fetchall()
        txs = db.execute("SELECT tx FROM fc_tx_history WHERE request=? ORDER BY at,rowid", (key,)).fetchall()
        risks = db.execute("SELECT flag FROM fc_risk WHERE partner=? AND resolved IS NULL", (row[0],)).fetchall()
        dispatch = db.execute("SELECT released FROM fc_dispatch WHERE request=?", (key,)).fetchone()
        sync = db.execute("SELECT state,successful FROM fc_stars_sync WHERE id=1").fetchone()
        blocked = db.execute("SELECT blocked FROM fc_partners WHERE id=?", (row[0],)).fetchone()[0]
        debt = f.balances(db, row[0])[3]
        threshold = f.number(db, "REFERRAL_LARGE_PAYOUT_USDT", 100, 10, 1000000)*f.UNIT
    return row, actions, txs, risks, dispatch, sync, blocked, debt, threshold


def payout_report(key, info=None):
    row, actions, txs, risks, dispatch, sync, blocked, debt, _ = info or _payout_info(key)
    partner, units, address, at, due, status, tx = row
    lines = [f"Заявка №{key} · партнёр {partner} · {f.display(units)} USDT · {status}",
             f"Создана {_date(at)} · срок {_date(due)}", f"USDT · TON Mainnet\n{address}",
             f"TXID {tx}" if tx else "TXID пока не зарегистрирован.",
             f"Сверка Stars · {_date(sync[1])} · {sync[0]}" if sync else "Сверка Stars ещё не выполнялась.",
             "Отметки риска · " + (", ".join(flag for (flag,) in risks) or "нет") +
             (" · партнёр заблокирован" if blocked else "") + (f" · долг {f.display(debt)} USDT" if debt else "")]
    if dispatch and status == 'submitted':
        lines.append(f"Перевод разрешён {_date(dispatch[0])}" if dispatch[0] else "Готовится. Перевод пока запрещён.")
    if len(txs) > 1:
        lines.append("Последние указанные TXID\n" + "\n".join(txid for (txid,) in txs[-5:]))
    return "\n".join(lines + [f"{_date(stamp)} · {event} · {reason}" for stamp, event, reason in actions])[:3900]


def command(actor, argument):
    action, _, args = argument.strip().partition(" ")
    if not action:
        return report()
    if action == "help":
        return ("🤝 /partners\n/partners user ID\n/partners payout №\n/partners block|unblock|clear-risk ID причина\n"
                "/partners approve|reject|dispatch № причина\n/partners tx № TXID\n/partners verify №\n"
                "/partners release № СУММА причина\n/partners reset № причина\n"
                "/partners replace № НОВЫЙ_TXID причина\n/partners confirm КОД\n/partners sync\n"
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
    if action == "sync":
        return f"✅ Сверка Stars завершена. Новых записей {stars.refresh()}."
    if action in ("release", "reset", "replace"):
        key, _, tail = args.partition(" ")
        value, _, reason = tail.partition(" ") if action != "reset" else ("", "", tail)
        token = payouts.start_intent(actor, _id(key), action, value, _notes(reason))
        return f"Подтверди операцию в течение пяти минут через /partners confirm {token}"
    if action == "confirm":
        return payouts.confirm_intent(actor, args)[1]
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
            return "Заявка подготовлена. Перед переводом получи одноразовое разрешение через /partners release или кнопку в карточке."
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
        txid, ref, count, nano, costs, charges = parts
        return f"✅ Сверка №{payouts.settle_batch(actor, txid, ref, int(count), int(nano), f.amount(costs), charges.split(','))}."
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


QUEUES = {"requested": "Новые заявки", "approved": "Одобренные заявки",
          "submitted": "Подготовка и проверка TON", "paid": "Завершённые выплаты"}
SETTINGS = {"REFERRAL_COMMISSION_PERCENT": ("Комиссия, %", 10),
            "MINIMUM_PAYOUT_USDT": ("Минимальный вывод, USDT", 10),
            "REFERRAL_HOLD_DAYS": ("Ожидание, дней", 21),
            "REFERRAL_PAYOUT_INTERVAL_DAYS": ("Время обработки, дней", 7),
            "WALLET_CHANGE_COOLDOWN_HOURS": ("Пауза после смены кошелька, ч", 48),
            "REFERRAL_LARGE_PAYOUT_USDT": ("Лимит крупной выплаты, USDT", 100),
            "REFERRAL_REGISTRATION_OPEN": ("Регистрация (0/1)", 1),
            "REFERRAL_ATTRIBUTION_OPEN": ("Атрибуция (0/1)", 1),
            "REFERRAL_FUNDING_OPEN": ("Обеспечение (0/1)", 1),
            "REFERRAL_WITHDRAWALS_OPEN": ("Вывод (0/1)", 1)}


@dataclass(frozen=True)
class Prompt:
    action: str
    target: str
    text: str


def _button(text, action):
    return {"text": text, "callback_data": "pa:"+action}


def _view(text, rows=(), back="home"):
    return text, {"inline_keyboard": [*rows, [_button("⬅️ Назад", back)]]}


def _queue(state, before):
    with storage.transaction() as db:
        f.schema(db)
        rows = db.execute("SELECT id,amount FROM fc_payout_requests WHERE status=? AND id<? ORDER BY id DESC LIMIT 9",
                          (state, before)).fetchall()
    buttons = [[_button(f"№{key} · {f.display(units)} USDT", f"card:{key}")] for key, units in rows[:8]]
    if len(rows) > 8:
        buttons.append([_button("Следующие", f"queue:{state}:{rows[7][0]}")])
    return _view(QUEUES[state] if rows else f"{QUEUES[state]} · пусто", buttons)


def _card(key, notice=""):
    info = _payout_info(key)
    row, _, _, _, dispatch, _, _, _, limit = info
    partner, units, address, _, _, status, tx = row
    buttons = [[_button("👤 Партнёр", f"user:{partner}")]]
    if status == "requested":
        buttons.append([_button("✅ Одобрить", f"approve:{key}"), _button("Отклонить", f"reject:{key}")])
    elif status == "approved":
        buttons += [[_button("Подготовить перевод", f"dispatch:{key}")], [_button("Отклонить", f"reject:{key}")]]
    elif status == "submitted" and dispatch and dispatch[0] is None:
        title = "Подтвердить крупную выплату" if units >= limit else "Разрешить ручной перевод"
        buttons += [[_button(title, f"release:{key}")], [_button("Отменить подготовку", f"reset:{key}")]]
    elif status == "submitted":
        buttons.append([{"text": "Копировать реквизиты", "copy_text": {"text": f"{address}\n{f.display(units)} USDT · TON Mainnet"}}])
        buttons.append([_button("Исправить TXID" if tx else "Ввести TXID", f"{'replace' if tx else 'tx'}:{key}")])
        if tx:
            buttons.append([_button("Проверить TON", f"verify:{key}")])
    return _view((notice+"\n\n"+payout_report(key, info))[:4000] if notice else payout_report(key, info),
                 buttons, f"queue:{status}" if status in QUEUES else "home")


def panel(actor, data):
    action, _, tail = data.partition(":")
    target, _, cursor = tail.partition(":")
    if action == "home":
        return "🤝 Партнёрская программа", {"inline_keyboard": [
            [_button("📊 Статистика", "stats"), _button("👥 Партнёры", "users")],
            *[[_button(label, f"queue:{state}")] for state, label in QUEUES.items()],
            [_button("⭐ Сверка Stars", "reconcile"), _button("💎 Резерв USDT", "treasury")],
            [_button("📋 Финансовый журнал", "ledger"), _button("⚙️ Настройки", "settings")],
            [{"text": "⬅️ Назад", "callback_data": "adm:home"}]]}
    if action == "stats":
        return _view(report())
    if action == "queue" and target in QUEUES:
        return _queue(target, _id(cursor) if cursor else 2**63-1)
    if action == "card":
        return _card(_id(target))
    if action == "verify":
        key = _id(target)
        try:
            payouts.verify_payout(actor, key)
        except f.Error as error:
            return _card(key, str(error))
        return _card(key, "✅ Перевод подтверждён.")
    if action == "confirm":
        key, result = payouts.confirm_intent(actor, target)
        return _card(key, result)
    if action in ("users", "ledger"):
        before = _id(target) if target else 2**63-1
        with storage.transaction() as db:
            f.schema(db)
            if action == "users":
                rows = db.execute("SELECT id,blocked FROM fc_partners WHERE id<? ORDER BY id DESC LIMIT 9", (before,)).fetchall()
            else:
                rows = db.execute("SELECT rowid,event,ref,partner,at,available,reserved,paid,debt FROM fc_referral_ledger "
                                  "WHERE rowid<? ORDER BY rowid DESC LIMIT 9", (before,)).fetchall()
        if action == "users":
            buttons = [[_button(f"{key}" + (" · заблокирован" if blocked else ""), f"user:{key}")]
                       for key, blocked in rows[:8]]
            buttons.insert(0, [_button("Найти по Telegram ID", "find")])
            text = "👥 Партнёры"
        else:
            buttons = [[_button("Аудит действий", "audit")]]
            text = "📋 Проводки USDT\n" + "\n".join(
                f"{_date(at)} · {event} · {partner} · {ref}\nДоступно {f.display(a)} · резерв {f.display(r)} · "
                f"выплачено {f.display(p)} · долг {f.display(d)}" for _, event, ref, partner, at, a, r, p, d in rows[:8])
        if len(rows) > 8:
            buttons.append([_button("Следующие", f"{action}:{rows[7][0]}")])
        return _view(text, buttons)
    if action == "user":
        key = _id(target)
        return _view(report(key), [[_button("Блокировать", f"block:{key}"), _button("Разблокировать", f"unblock:{key}")],
                                   [_button("Снять отметки риска", f"clear-risk:{key}")]], "users")
    if action in ("reconcile", "sync"):
        notice = f"Новых записей {stars.refresh()}.\n\n" if action == "sync" else ""
        return _view(notice+reconciliation_report(), [[_button("🔄 Сверить сейчас", "sync")],
                                                      [_button("Провести сверку и обеспечение", "batch")]])
    if action == "treasury":
        return _view(treasury_report(), [[_button("Учесть пополнение", "topup")]])
    if action == "audit":
        return _view(command(actor, "audit"), back="ledger")
    if action == "settings":
        with storage.transaction() as db:
            f.schema(db)
            buttons = [[_button(f"{label} · {f.get_setting(db,key,default)}", f"set:{key}")]
                       for key, (label, default) in SETTINGS.items()]
        return _view("⚙️ Настройки партнёрской программы", buttons + [[_button("Справочный курс Stars", "rate")]])
    prompts = {"tx": "Введи TXID перевода.", "replace": "Введи новый TXID и причину через пробел.",
               "release": "Введи точную сумму USDT и причину через пробел.", "find": "Введи Telegram ID партнёра.",
               "topup": "Введи TXID пополнения и причину через пробел.",
               "batch": "Введи TXID FRAGMENT_ID STARS TON_NANO COST_USDT CHARGE[,CHARGE].",
               "rate": "Введи курс в микро-USDT за одну Star и источник через пробел.",
               "set": "Введи новое значение и причину через пробел."}
    if action in prompts or action in ("approve", "reject", "dispatch", "reset", "block", "unblock", "clear-risk"):
        if action == "set" and target not in SETTINGS:
            raise f.Error("Неизвестная настройка.")
        if action == "release":
            info = _payout_info(_id(target))
            if info[0][1] < info[-1]:
                return Prompt(action, target, "Укажи причину разрешения ручного перевода.")
        return Prompt(action, target, prompts.get(action, "Укажи причину операции."))
    raise f.Error("Действие недоступно. Открой партнёрскую программу заново.")


def panel_reply(actor, action, target, text):
    if action == "find":
        return panel(actor, f"user:{_id(text)}")
    if action in ("release", "reset", "replace"):
        key = _id(target)
        value, reason = "", text
        if action == "replace" or (action == "release" and (info := _payout_info(key))[0][1] >= info[-1]):
            value, _, reason = text.partition(" ")
        token = payouts.start_intent(actor, key, action, value, _notes(reason))
        label = {"replace": f"Подтверди замену TXID на {value}.", "reset": "Подтверди отмену подготовки.",
                 "release": "Подтверди проверку перед однократным разрешением перевода."}[action]
        return _view(label, [[_button("✅ Подтвердить", f"confirm:{token}")]], f"card:{key}")
    result = command(actor, " ".join(part for part in (action, target, text) if part))
    if action in ("tx", "approve", "reject", "dispatch"):
        return _card(_id(target), result)
    return _view(result, back="settings" if action in ("rate", "set") else "home")
