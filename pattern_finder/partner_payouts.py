"""Wallet binding and reviewed financial operations for Finder Candles Partners."""
import os
import re
import secrets

import mini_star_audit as stars
import partner_finance as f
import partner_ton as ton
import storage


def _open(db, partner):
    return db.execute("SELECT id FROM fc_payout_requests WHERE partner=? "
                      "AND status IN ('requested','approved','submitted')", (partner,)).fetchone()


def _partner(db, user):
    if not db.execute("SELECT 1 FROM fc_partners WHERE id=?", (user,)).fetchone():
        raise f.Error("Сначала получи партнёрскую ссылку.")


def challenge(user, now=None):
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        _partner(db, user)
        if _open(db, user):
            raise f.Error("Закрой текущую заявку перед сменой кошелька.")
        token = secrets.token_urlsafe(32)
        db.execute("INSERT OR REPLACE INTO fc_ton_challenges VALUES (?,?,?,?)",
                   (user, token, now, now + ton.PROOF_TTL))
        return token


def bind(user, account, proof, domain, now=None):
    use_clock = now is None
    now = f.timestamp(now)
    if not isinstance(account, dict) or not isinstance(proof, dict) or not isinstance(proof.get("payload"), str):
        raise f.Error("Кошелёк не передал ton_proof.")
    token = proof["payload"]
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT issued FROM fc_ton_challenges WHERE partner=? AND token=? AND expires>?",
                         (user, token, now)).fetchone()
        if not row:
            raise f.Error("Подтверждение истекло. Подключи кошелёк заново.")
    wallet = ton.proof_owner(account, proof, domain, row[0], now)
    with storage.transaction() as db:
        if _open(db, user):
            raise f.Error("Смена кошелька недоступна при открытой заявке.")
        deadline = f.timestamp() if use_clock else now
        if not db.execute("DELETE FROM fc_ton_challenges WHERE partner=? AND token=? AND expires>?",
                          (user, token, deadline)).rowcount:
            raise f.Error("Подтверждение уже использовано или истекло.")
        old = db.execute("SELECT address,unlock,verified FROM fc_wallets WHERE partner=?", (user,)).fetchone()
        unlock = (now + 3600 * f.number(db, "WALLET_CHANGE_COOLDOWN_HOURS", 48, 48, 720)
                  if old and old[0] != wallet else old[1] if old else now)
        db.execute("INSERT OR REPLACE INTO fc_wallets VALUES (?,?,?,?,1,'TON_MAINNET')",
                   (user, wallet, now, unlock))
        shared = db.execute("SELECT partner FROM fc_wallets WHERE address=? AND verified=1", (wallet,)).fetchall()
        if len(shared) > 1:
            for (partner,) in shared:
                f.risk(db, partner, "shared_wallet", now)
        f.audit(db, user, wallet, "wallet_verified", "TON Connect", now)
        f.notify(db, user, ("💎 Адрес изменён. Вывод временно ограничен." if unlock > now
                            else "💎 TON-кошелёк подтверждён."))


def disconnect(user):
    now = f.timestamp()
    with storage.transaction() as db:
        f.schema(db)
        if _open(db, user):
            raise f.Error("Смена кошелька недоступна при открытой заявке.")
        db.execute("DELETE FROM fc_ton_challenges WHERE partner=?", (user,))
        if db.execute("UPDATE fc_wallets SET verified=0 WHERE partner=? AND verified=1", (user,)).rowcount:
            f.audit(db, user, user, "wallet_disconnected", "По запросу партнёра", now)


def _withdrawals(db, now):
    if (not f.enabled(now) or f.flag("REFERRAL_TEST_MODE") or not f.flag("REFERRAL_PAYOUTS_ENABLED")
            or f.get_setting(db, "REFERRAL_WITHDRAWALS_OPEN", 1) != "1"):
        raise f.Error("Вывод средств пока недоступен.")


def _binding(db, owner):
    row = db.execute("SELECT value FROM fc_settings WHERE key='treasury_binding'").fetchone()
    if row and row[0] != owner:
        raise f.Error("Кошелёк резерва изменён. Нужна сверка обязательств.")
    if not row:
        db.execute("INSERT INTO fc_settings VALUES ('treasury_binding',?)", (owner,))


def verify_reserve(db, owner, live, extra=0):
    _binding(db, owner)
    available, reserved, paid, _ = f.balances(db)
    received = db.execute("SELECT COALESCE(SUM(amount),0) FROM fc_treasury_ledger").fetchone()[0]
    if min(live, received-paid) < available+reserved+extra:
        raise f.Error("Средства резерва требуют сверки. Попробуй позже.")


def request(user, units, admin_ids=(), now=None):
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        _withdrawals(db, now)
        _partner(db, user)
        if units < f.number(db, "MINIMUM_PAYOUT_USDT", 10, 10, 1000000) * f.UNIT:
            raise f.Error("Сумма меньше минимальной для вывода.")
        if units > f.balances(db, user)[0] or _open(db, user):
            raise f.Error("Недостаточно доступных средств или уже есть открытая заявка.")
    owner = ton.treasury()
    _, live = ton.jetton_wallet(owner)
    with storage.transaction() as db:
        _withdrawals(db, now)
        if (_open(db, user) or units > f.balances(db, user)[0]
                or units < f.number(db, "MINIMUM_PAYOUT_USDT", 10, 10, 1000000) * f.UNIT):
            raise f.Error("Баланс или условия изменились. Обнови экран.")
        if db.execute("SELECT blocked FROM fc_partners WHERE id=?", (user,)).fetchone()[0] or f.balances(db, user)[3]:
            raise f.Error("Вывод требует проверки.")
        wallet = db.execute("SELECT address,unlock FROM fc_wallets WHERE partner=? AND verified=1",
                            (user,)).fetchone()
        if not wallet or wallet[1] > now:
            raise f.Error("Подтверди кошелёк и дождись окончания ограничения на вывод.")
        if wallet[0] == owner:
            raise f.Error("Кошелёк резерва нельзя использовать для заявки.")
        verify_reserve(db, owner, live)
        due = now + f.number(db, "REFERRAL_PAYOUT_INTERVAL_DAYS", 7, 1, 30) * f.DAY
        key = db.execute("""INSERT INTO fc_payout_requests(partner,amount,address,source,at,due,status)
                            VALUES (?,?,?,?,?,?,'requested')""", (user, units, wallet[0], owner, now, due)).lastrowid
        f.journal(db, "reserve", key, user, now, available=-units, reserved=units)
        f.audit(db, user, key, "requested", "Зарезервировано", now)
        f.notify(db, user, f"💳 Заявка №{key} принята.")
        for admin in admin_ids:
            f.notify(db, admin, f"Партнёры: новая заявка №{key} на {f.display(units)} USDT.")
        return key


def decision(actor, key, approve, reason, now=None):
    now = f.timestamp(now)
    if approve:
        owner = ton.treasury()
        _, live = ton.jetton_wallet(owner)
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT partner,amount,status,source FROM fc_payout_requests WHERE id=?", (key,)).fetchone()
        if not row or row[2] not in (("requested",) if approve else ("requested", "approved")):
            raise f.Error("Состояние заявки изменилось.")
        partner, units, _, source = row
        if approve:
            _withdrawals(db, now)
            if (source != owner or db.execute("SELECT blocked FROM fc_partners WHERE id=?", (partner,)).fetchone()[0]
                    or db.execute("SELECT 1 FROM fc_risk WHERE partner=? AND resolved IS NULL", (partner,)).fetchone()
                    or f.balances(db, partner)[3]):
                raise f.Error("Сначала проверь ограничения и риски партнёра.")
            verify_reserve(db, owner, live)
        else:
            f.credit(db, "reject", key, partner, now, units, reserved=-units)
        status = "approved" if approve else "rejected"
        db.execute("UPDATE fc_payout_requests SET status=?,reason=? WHERE id=?", (status, reason, key))
        f.audit(db, actor, key, status, reason, now)
        f.notify(db, partner, f"Заявка №{key} одобрена." if approve else f"Заявка №{key} отклонена. {reason}")


def _preflight(db, key, status, owner, live, now):
    _withdrawals(db, now)
    stars.require_fresh(db, now)
    row = db.execute("SELECT partner,source,status,address,amount FROM fc_payout_requests WHERE id=?", (key,)).fetchone()
    if not row or row[1] != owner or row[2] != status:
        raise f.Error("Состояние заявки изменилось.")
    partner = row[0]
    if (db.execute("SELECT blocked FROM fc_partners WHERE id=?", (partner,)).fetchone()[0]
            or db.execute("SELECT 1 FROM fc_risk WHERE partner=? AND resolved IS NULL", (partner,)).fetchone()
            or f.balances(db, partner)[3]):
        raise f.Error("Сначала проверь ограничения и риски партнёра.")
    verify_reserve(db, owner, live)
    return row[3], row[4]


def dispatch(actor, key, reason, now=None):
    """Prepare a payout without authorizing movement of funds."""
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        stars.require_fresh(db, now)
    owner = ton.treasury()
    _, live = ton.jetton_wallet(owner)
    with storage.transaction() as db:
        _preflight(db, key, "approved", owner, live, now)
        db.execute("INSERT INTO fc_dispatch(request,actor,prepared,nonce) VALUES (?,?,?,?)",
                   (key, actor, now, secrets.token_urlsafe(18)))
        db.execute("UPDATE fc_payout_requests SET status='submitted',submitted=NULL WHERE id=?", (key,))
        f.audit(db, actor, key, "dispatch", reason, now)


def _intent_context(db, key, action, value):
    row = db.execute("""SELECT p.status,p.tx,p.amount,d.nonce,d.released FROM fc_payout_requests p
        LEFT JOIN fc_dispatch d ON d.request=p.id WHERE p.id=?""", (key,)).fetchone()
    if not row or row[0] != "submitted":
        raise f.Error("Заявка уже завершена или ещё не подготовлена.")
    if action == "replace":
        if not row[1] or row[1] == value:
            raise f.Error("Для исправления укажи другой TXID.")
        return row[1]
    if not row[3] or row[4] is not None or row[1]:
        raise f.Error("Перевод уже мог начаться. Резерв останется до сверки фактической транзакции.")
    if action == "release":
        limit = f.number(db, "REFERRAL_LARGE_PAYOUT_USDT", 100, 10, 1000000)*f.UNIT
        if row[2] >= limit and (not value or f.amount(value) != row[2]):
            raise f.Error(f"Крупная выплата. Введи точную сумму {f.display(row[2])} USDT для подтверждения.")
    return row[3]


def start_intent(actor, key, action, value, reason, now=None):
    if action not in ("release", "reset", "replace") or not reason.strip():
        raise f.Error("Нужна причина операции.")
    if action == "replace":
        value = ton.hash_id(value)
    now = f.timestamp(now)
    token = secrets.token_urlsafe(18)
    with storage.transaction() as db:
        f.schema(db)
        expected = _intent_context(db, key, action, value)
        db.execute("DELETE FROM fc_operator_intents WHERE expires<=? OR (actor=? AND request=? AND action=?)",
                   (now, actor, key, action))
        db.execute("INSERT INTO fc_operator_intents VALUES (?,?,?,?,?,?,?,?)",
                   (token, actor, key, action, expected, value, reason[:200], now+300))
        if action == "replace":
            f.audit(db, actor, key, "tx_replacement_proposed", f"{value} · {reason[:200]}", now)
    return token


def _intent(db, actor, token, now):
    row = db.execute("SELECT request,action,expected,value,reason FROM fc_operator_intents "
                     "WHERE token=? AND actor=? AND expires>?", (token, actor, now)).fetchone()
    if not row or _intent_context(db, row[0], row[1], row[3]) != row[2]:
        raise f.Error("Подтверждение истекло, уже использовано или заявка изменилась.")
    return row


def confirm_intent(actor, token, now=None):
    check_time = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        intent = _intent(db, actor, token, check_time)
        key, action, expected, value, reason = intent
        payment = db.execute("SELECT source,address,amount,submitted FROM fc_payout_requests WHERE id=?", (key,)).fetchone()
    if action == "release":
        stars.refresh(now=now)
        owner = ton.treasury()
        _, live = ton.jetton_wallet(owner)
    elif action == "replace":
        source, address, amount, submitted = payment
        replacement = ton.verify_transfer(value, source, address, amount, submitted)
        try:
            ton.verify_transfer(expected, source, address, amount, submitted)
        except f.Error:
            pass
        else:
            raise f.Error("Текущий TXID уже подтверждается. Проверь его и сверь перевод вручную.")
    check_time = f.timestamp(now)
    with storage.transaction() as db:
        if _intent(db, actor, token, check_time) != intent:
            raise f.Error("Заявка изменилась. Обнови карточку.")
        if action == "release":
            address, amount = _preflight(db, key, "submitted", owner, live, check_time)
            db.execute("UPDATE fc_dispatch SET released=?,released_by=? WHERE request=?", (check_time, actor, key))
            db.execute("UPDATE fc_payout_requests SET submitted=? WHERE id=?", (check_time, key))
            f.audit(db, actor, key, "release", reason, check_time)
            result = f"Перевод разрешён один раз. Отправь {f.display(amount)} USDT в TON Mainnet на {address}. Сразу запиши TXID."
        elif action == "reset":
            db.execute("DELETE FROM fc_dispatch WHERE request=?", (key,))
            db.execute("UPDATE fc_payout_requests SET status='approved' WHERE id=?", (key,))
            f.audit(db, actor, key, "reset_preparation", reason, check_time)
            result = "Подготовка отменена до разрешения перевода. Резерв сохранён."
        else:
            if _accounted(db, replacement.root, replacement.receipt):
                raise f.Error("Фактический перевод уже учтён.")
            db.execute("INSERT OR IGNORE INTO fc_tx_history VALUES (?,?,NULL,?,'Историческая запись')",
                       (expected, key, check_time))
            _record_tx(db, actor, key, value, reason, check_time)
            result = "TXID исправлен. Запусти проверку выплаты."
        db.execute("DELETE FROM fc_operator_intents WHERE token=?", (token,))
    return key, result


def _record_tx(db, actor, key, txid, reason, now):
    if (db.execute("SELECT 1 FROM fc_tx_history WHERE tx=? UNION ALL SELECT 1 FROM fc_payout_requests WHERE tx=?",
                   (txid, txid)).fetchone() or _accounted(db, txid, txid)):
        raise f.Error("TXID уже использован.")
    db.execute("INSERT INTO fc_tx_history VALUES (?,?,?,?,?)", (txid, key, actor, now, reason))
    db.execute("UPDATE fc_payout_requests SET tx=? WHERE id=?", (txid, key))
    f.audit(db, actor, key, "tx_submitted", txid, now)


def _accounted(db, root, receipt):
    return db.execute("SELECT 1 FROM fc_payout_transactions WHERE tx IN (?,?) OR destination_tx IN (?,?) "
                      "UNION ALL SELECT 1 FROM fc_treasury_ledger WHERE tx IN (?,?) OR destination_tx IN (?,?)",
                      (root, receipt)*4).fetchone()


def submit(actor, key, txid, now=None):
    now = f.timestamp(now)
    txid = ton.hash_id(txid)
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT status,tx,partner,submitted FROM fc_payout_requests WHERE id=?", (key,)).fetchone()
        if not row or row[0] != "submitted" or row[3] is None or (row[1] and row[1] != txid):
            raise f.Error("Заявка должна быть передана на выплату. Указанный TXID закрепляется за ней.")
        if row[1] == txid:
            return
        _record_tx(db, actor, key, txid, "Первичный ввод", now)
        f.notify(db, row[2], f"⏳ Для заявки №{key} зарегистрирован перевод. Ожидаем подтверждения TON.")


def verify_payout(actor, key, now=None):
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        row = db.execute("SELECT partner,amount,address,source,tx,submitted,status FROM fc_payout_requests WHERE id=?",
                         (key,)).fetchone()
        if not row or not row[4] or row[6] != "submitted":
            raise f.Error("Укажи TXID перед проверкой.")
    user, units, destination, source, tx, submitted, _ = row
    try:
        transfer = ton.verify_transfer(tx, source, destination, units, submitted)
    except f.Error as error:
        with storage.transaction() as db:
            if db.execute("SELECT status,tx FROM fc_payout_requests WHERE id=?", (key,)).fetchone() == ("submitted", tx):
                f.audit(db, actor, key, "verify_failed", str(error), now)
        raise
    with storage.transaction() as db:
        if db.execute("SELECT status,tx FROM fc_payout_requests WHERE id=?", (key,)).fetchone() != ("submitted", tx):
            raise f.Error("Состояние заявки уже изменилось.")
        if _accounted(db, transfer.root, transfer.receipt):
            raise f.Error("Перевод уже учтён.")
        db.execute("INSERT INTO fc_payout_transactions VALUES (?,?,?,?)",
                   (transfer.root, key, transfer.receipt, transfer.at))
        f.journal(db, "paid", key, user, now, reserved=-units, paid=units)
        db.execute("UPDATE fc_payout_requests SET status='paid' WHERE id=?", (key,))
        f.audit(db, actor, key, "paid", transfer.root, now)
        f.notify(db, user, f"✅ Выплата №{key} на {f.display(units)} USDT подтверждена в TON.")


def settle_batch(actor, txid, source_ref, stars, ton_nano, costs, charges, now=None):
    """Record an owner-attested Fragment conversion backed by an independently verified inflow."""
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        if not f.funding_enabled(db, now):
            raise f.Error("Обеспечение начислений пока отключено.")
    if not (0 < stars <= 10**8 and 0 < ton_nano <= 10**17 and 0 <= costs <= 10**14
            and re.fullmatch(r"[A-Za-z0-9_-]{6,100}", source_ref) and 0 < len(charges) <= 100
            and len(set(charges)) == len(charges)):
        raise f.Error("Укажи уникальную ссылку на запись Fragment, суммы и список платежей.")
    owner = ton.treasury()
    source = ton.address(os.getenv("TON_SETTLEMENT_SOURCE_ADDRESS", ""))
    if owner == source:
        raise f.Error("Источник пополнения должен отличаться от кошелька выплат.")
    transfer = ton.verify_transfer(txid, source, owner, since=f.cutover())
    _, live = ton.jetton_wallet(owner)
    proceeds = transfer.amount - costs
    if proceeds <= 0 or not now - 30*f.DAY <= transfer.at <= now:
        raise f.Error("Пополнение устарело или расходы превышают поступление.")
    with storage.transaction() as db:
        f.schema(db)
        if not f.funding_enabled(db, now):
            raise f.Error("Обеспечение начислений пока отключено.")
        _binding(db, owner)
        if (db.execute("SELECT 1 FROM fc_stars_batches WHERE source_ref=?", (source_ref,)).fetchone()
                or _accounted(db, transfer.root, transfer.receipt)):
            raise f.Error("Пополнение или сверка уже учтены.")
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='mini_star_audit'").fetchone():
            raise f.Error("Сначала импортируй официальный журнал Stars.")
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='fc_fragment_withdrawals'").fetchone():
            raise f.Error("Сначала импортируй официальный журнал Fragment.")
        withdrawal = db.execute("SELECT stars,confirmed FROM fc_fragment_withdrawals WHERE id=?", (source_ref,)).fetchone()
        if not withdrawal or withdrawal[0] != stars or not transfer.at >= withdrawal[1] > f.cutover():
            raise f.Error("Нет подтверждённого вывода Fragment на указанную сумму Stars.")
        items = []
        for charge in charges:
            row = db.execute("""SELECT c.partner,c.stars,c.rate_bps FROM fc_referral_commissions c
                JOIN payments p ON p.id=c.charge JOIN mini_star_audit a ON a.charge=c.charge AND a.event='payment'
                WHERE c.charge=? AND c.mature<=? AND p.status='succeeded'
                AND a.ledger_nanos=c.stars*1000000000 AND COALESCE(a.commission_nanos,0)=0
                AND NOT EXISTS(SELECT 1 FROM mini_star_audit ar WHERE ar.charge=c.charge AND ar.event='refund')
                AND NOT EXISTS(SELECT 1 FROM fc_reversals r WHERE r.charge=c.charge)
                AND NOT EXISTS(SELECT 1 FROM fc_allocations x WHERE x.charge=c.charge)""",
                             (charge, withdrawal[1])).fetchone()
            if not row:
                raise f.Error("В наборе есть непроверенный, возвращённый или уже обеспеченный платёж.")
            items.append((charge, *row))
        if sum(row[2] for row in items) > stars:
            raise f.Error("Покупок больше, чем выведено Stars.")
        db.execute("INSERT INTO fc_treasury_ledger VALUES (?,?,?,?,?,?)",
                   (transfer.root, transfer.receipt, transfer.amount, transfer.at, source, owner))
        batch = db.execute("INSERT INTO fc_stars_batches(tx,source_ref,stars,ton_nano,usdt,costs,at) VALUES (?,?,?,?,?,?,?)",
                           (transfer.root, source_ref, stars, ton_nano, transfer.amount, costs, now)).lastrowid
        for charge, user, bought, bps in items:
            value = proceeds * bought * bps // (stars * 10000)
            before = f.balances(db, user)[0]
            db.execute("INSERT INTO fc_allocations VALUES (?,?,?)", (charge, batch, value))
            f.credit(db, "fund", charge, user, now, value)
            f.notify(db, user, f"✅ Начисление обеспечено. Комиссия {f.display(value)} USDT.")
            threshold = f.number(db, "MINIMUM_PAYOUT_USDT", 10, 10, 1000000) * f.UNIT
            if before < threshold <= f.balances(db, user)[0]:
                f.notify(db, user, "💳 Баланс достиг минимальной суммы вывода.")
        verify_reserve(db, owner, live)
        f.audit(db, actor, batch, "settlement", source_ref, now)
        return batch


def record_topup(actor, txid, reason):
    now = f.timestamp()
    with storage.transaction() as db:
        f.schema(db)
        if not f.funding_enabled(db, now):
            raise f.Error("Финансовая сверка пока отключена.")
    owner = ton.treasury()
    source = ton.address(os.getenv("TON_SETTLEMENT_SOURCE_ADDRESS", ""))
    if source == owner:
        raise f.Error("Источник пополнения должен отличаться от кошелька выплат.")
    transfer = ton.verify_transfer(txid, source, owner, since=f.cutover())
    with storage.transaction() as db:
        f.schema(db)
        _binding(db, owner)
        if not f.funding_enabled(db, now):
            raise f.Error("Финансовая сверка пока отключена.")
        if _accounted(db, transfer.root, transfer.receipt):
            raise f.Error("Перевод уже учтён.")
        db.execute("INSERT INTO fc_treasury_ledger VALUES (?,?,?,?,?,?)",
                   (transfer.root, transfer.receipt, transfer.amount, transfer.at, source, owner))
        f.audit(db, actor, transfer.root, "treasury_topup", reason, now)
    return transfer.amount


def overview(user, now=None):
    now = f.timestamp(now)
    with storage.transaction() as db:
        f.schema(db)
        allowed = f.registration_enabled(db, now)
        attribution = f.attribution_enabled(db, now)
        funding = f.funding_enabled(db, now)
        wallet = db.execute("SELECT address,unlock FROM fc_wallets WHERE partner=? AND verified=1", (user,)).fetchone()
        code = db.execute("SELECT code FROM fc_referral_codes WHERE partner=?", (user,)).fetchone()
        invited = db.execute("SELECT COUNT(*) FROM fc_referral_attributions WHERE partner=?", (user,)).fetchone()[0]
        purchases = db.execute("SELECT COUNT(*) FROM fc_referral_commissions WHERE partner=?", (user,)).fetchone()[0]
        invitations = db.execute("""SELECT referral.at, CASE WHEN r.charge IS NOT NULL THEN 'reversed'
            WHEN c.charge IS NOT NULL THEN 'purchased' ELSE 'invited' END
            FROM fc_referral_attributions referral LEFT JOIN fc_referral_commissions c ON c.buyer=referral.buyer
            LEFT JOIN fc_reversals r ON r.charge=c.charge
            WHERE referral.partner=? ORDER BY referral.at DESC LIMIT 20""", (user,)).fetchall()
        states = db.execute("""SELECT CASE WHEN r.charge IS NOT NULL THEN 'reversed' WHEN a.charge IS NOT NULL THEN 'funded'
            WHEN c.mature>? THEN 'pending' ELSE 'matured_unfunded' END AS state, COUNT(*)
            FROM fc_referral_commissions c LEFT JOIN fc_allocations a USING(charge)
            LEFT JOIN fc_reversals r USING(charge) WHERE c.partner=? GROUP BY state""", (now, user)).fetchall()
        estimate, unquoted = db.execute("""SELECT SUM(c.estimate),COUNT(*)-COUNT(c.estimate)
            FROM fc_referral_commissions c LEFT JOIN fc_allocations a USING(charge)
            LEFT JOIN fc_reversals r USING(charge) WHERE c.partner=? AND a.charge IS NULL AND r.charge IS NULL""",
            (user,)).fetchone()
        earnings = db.execute("""SELECT c.plan,c.at,CASE WHEN r.charge IS NOT NULL THEN 'reversed'
            WHEN a.charge IS NOT NULL THEN 'funded' WHEN c.mature>? THEN 'pending' ELSE 'matured_unfunded' END,
            a.amount FROM fc_referral_commissions c LEFT JOIN fc_allocations a USING(charge)
            LEFT JOIN fc_reversals r USING(charge) WHERE c.partner=? ORDER BY c.at DESC LIMIT 20""", (now, user)).fetchall()
        payouts = db.execute("""SELECT id,amount,status,due,
            CASE WHEN status IN ('rejected','cancelled') THEN reason END
            FROM fc_payout_requests WHERE partner=? ORDER BY id DESC LIMIT 20""", (user,)).fetchall()
        available, reserved, paid, debt = f.balances(db, user)
        percent = f.number(db, "REFERRAL_COMMISSION_PERCENT", 10, 1, 100)
        minimum = f.number(db, "MINIMUM_PAYOUT_USDT", 10, 10, 1000000)*f.UNIT
        blocked = db.execute("SELECT blocked FROM fc_partners WHERE id=?", (user,)).fetchone()
        payout_enabled = (f.enabled(now) and f.flag("REFERRAL_PAYOUTS_ENABLED") and not f.flag("REFERRAL_TEST_MODE")
                          and f.get_setting(db, "REFERRAL_WITHDRAWALS_OPEN", 1) == "1")
    funded = available
    if available:
        try:
            owner = ton.treasury()
            _, live = ton.jetton_wallet(owner)
            with storage.transaction() as db:
                verify_reserve(db, owner, live)
                funded = f.balances(db, user)[0]
        except f.Error:
            funded = 0
    return {"enabled": allowed, "wallet_connect": f.flag("REFERRALS_ENABLED"), "attribution_open": attribution,
            "funding_open": funding, "commission": percent,
            "registered": bool(code), "link": "https://t.me/findercandles_bot?start=fc_"+code[0] if code else None,
            "invited": invited, "invitations": invitations, "purchases": purchases, "counts": dict(states),
            "earnings": [(p, at, state, str(n) if n is not None else None) for p, at, state, n in earnings],
            "payouts": [(key, str(n), state, due, reason) for key, n, state, due, reason in payouts],
            "wallet": {"address": wallet[0], "unlock": wallet[1]} if wallet else None,
            "available": f.display(funded), "held": f.display(max(0, available-funded)), "reserved": f.display(reserved),
            "estimated": f.display(estimate) if estimate is not None else None, "unquoted": unquoted,
            "paid": f.display(paid), "debt": f.display(debt), "minimum": f.display(minimum),
            "remaining": f.display(max(0, minimum-funded)),
            "can_request": bool(payout_enabled and not (blocked and blocked[0]) and not debt and funded >= minimum
                                and wallet and wallet[1] <= now and not any(r[2] in f.ACTIVE for r in payouts)),
            "manifest": os.getenv("TON_CONNECT_MANIFEST_URL", "")}
