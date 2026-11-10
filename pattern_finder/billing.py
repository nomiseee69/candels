"""Подписка: тарифы Free / Pro, оплата Telegram Stars и ЮKassa, лимиты, напоминания.

Тарифы
  Free — 2 паттерна, ручной скан 3 раза в день, без автоскана и без нейросети-мозга.
  Pro  — всё без ограничений. 30 дней или год (со скидкой).
  Новичку — 3 дня Pro бесплатно (один раз на аккаунт Telegram), потом сам переходит на Free.

Защита от обмана
  • Счета звёздами создаёт только сам бот (нужен его токен), в счёте записаны план и цена
    («pro_month_500»). Перед списанием Telegram спрашивает бота (pre_checkout) — бот проверяет
    план, валюту XTR и что сумма = цене в счёте; после оплаты проверяет ещё раз.
  • Сообщение «оплачено» (successful_payment) присылает только Telegram — подделать его
    пользователь не может. Один и тот же платёж (charge_id) засчитывается один раз.
  • Возврат звёзд (refunded_payment) — Pro снимается сразу.
  • ЮKassa: бот сам спрашивает у ЮKassa статус платежа по его id (не верит пользователю) и
    проверяет: оплачен, сумма и валюта = цене плана, платёж создан для этого же пользователя,
    не тестовый (кроме YOOKASSA_ALLOW_TEST=1). Один платёж — один раз.
  Админ (ADMIN_IDS в .env, по умолчанию TELEGRAM_CHAT_ID) — всегда Pro.

Оплата
  ⭐ Telegram Stars — прямо в чате. Месяц — подписка, Telegram сам продлевает её каждые 30 дней;
     год — разовый платёж. Это способ, который Telegram требует для цифровых товаров.
  💳 ЮKassa — ссылка на страницу оплаты картой РФ / СБП. Бот раз в минуту спрашивает ЮKassa,
     прошла ли оплата (сервер с публичным адресом и webhook не нужны).
     ⚠ Telegram требует для цифровых товаров Stars; оплата сторонней кассой — риск, что Telegram
     скроет бота в мобильных приложениях. Решение принято сознательно; выключить —
     убрать YOOKASSA_SHOP_ID из .env.

Настройки в .env (все необязательные, кроме ключей ЮKassa для оплаты картой):
  ADMIN_IDS=123,456            STARS_MONTH=500   STARS_YEAR=4800
  RUB_MONTH=990  RUB_YEAR=9490 TRIAL_DAYS=3
  YOOKASSA_SHOP_ID=…  YOOKASSA_SECRET_KEY=…  YOOKASSA_RETURN_URL=https://t.me/имя_бота
"""

import json
import os
import time
import uuid

import requests

import storage

DAY = 86400
MONTH_SECONDS = 30 * DAY            # Telegram принимает у подписки только период 30 дней


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


PLANS = {   # план: (дней, звёзд, рублей, название)
    "month": (30, _int("STARS_MONTH", 500), _int("RUB_MONTH", 990), "Pro на 30 дней"),
    "year": (365, _int("STARS_YEAR", 4800), _int("RUB_YEAR", 9490), "Pro на год (−20%)"),
}
USD = {"month": 10, "year": 96}     # для текста: ≈ $10 / ≈ $96
TRIAL_DAYS = _int("TRIAL_DAYS", 3)
FREE = {"patterns": 2, "scans": 3}  # Free: паттернов, ручных сканов в день
REMIND_DAYS = 3                     # напомнить о продлении за 3 дня


def admins():
    raw = os.getenv("ADMIN_IDS") or os.getenv("TELEGRAM_CHAT_ID") or ""
    return {int(x) for x in raw.replace(" ", "").split(",") if x.lstrip("-").isdigit()}


def is_admin(chat_id):
    return int(chat_id) in admins()


# --- статус -----------------------------------------------------------------------------------------
def _sub(user):
    return user.setdefault("sub", {})


def status(chat_id, now=None):
    """{"pro", "until", "source", "trial", "admin"}. Пользователь без пробного — получает его
    при первом обращении (и новички, и те, кто пользовался ботом до подписки)."""
    now = now or time.time()
    if is_admin(chat_id):
        return {"pro": True, "until": None, "source": "admin", "trial": False, "admin": True}

    sub = storage.get_user(chat_id).get("sub") or {}
    if sub.get("forever"):                              # админ выдал Pro без срока
        return {"pro": True, "until": None, "source": "forever", "trial": False, "admin": False,
                "recurring": False, "forever": True}
    if not sub.get("trial_used"):                       # пишем в базу только один раз — при выдаче пробного

        def give_trial(user):
            sub = _sub(user)
            if not sub.get("trial_used"):
                sub.update(trial_used=True, until=max(sub.get("until") or 0, now + TRIAL_DAYS * DAY),
                           source="trial", trial_started=now)
            return dict(sub)
        sub = storage.update_user(chat_id, give_trial)
    elif sub.get("source") == "trial" and sub.get("until"):
        # пробный не длиннее TRIAL_DAYS (в обн. 40 он был 7 дней — укорачиваем выданные тогда)
        started = sub.get("trial_started") or sub["until"] - 7 * DAY
        limit = started + TRIAL_DAYS * DAY
        if sub["until"] > limit + 60:

            def shorten(user):
                _sub(user).update(until=limit, trial_started=started)
                return dict(_sub(user))
            sub = storage.update_user(chat_id, shorten)
    until = sub.get("until") or 0
    return {"pro": until > now, "until": until or None, "source": sub.get("source"),
            "trial": sub.get("source") == "trial", "admin": False, "recurring": sub.get("recurring", False)}


def is_pro(chat_id, now=None):
    return status(chat_id, now)["pro"]


def extend(chat_id, days=None, source="admin", until=None, recurring=False, now=None):
    """Продлить Pro: на days дней от конца текущей подписки (или от сейчас) либо до until."""
    now = now or time.time()

    def change(user):
        sub = _sub(user)
        sub["trial_used"] = True
        base = max(sub.get("until") or 0, now)
        new = base + (days or 0) * DAY
        if until:
            new = max(new, until)                       # подписка звёздами: дату окончания называет Telegram
        sub["until"] = new
        sub.update(source=source, recurring=recurring, reminded=None, expired_told=False)
        return sub["until"]
    return storage.update_user(chat_id, change)


def grant_forever(chat_id):
    """Pro навсегда (подарок от админа): без даты окончания и без напоминаний."""
    def change(user):
        sub = _sub(user)
        sub.update(forever=True, trial_used=True, source="forever", recurring=False, reminded=None,
                   expired_told=False)
        return True
    return storage.update_user(chat_id, change)


def revoke(chat_id, now=None):
    now = now or time.time()
    return storage.update_user(chat_id, lambda u: _sub(u).update(until=now - 1, recurring=False, trial_used=True,
                                                                 forever=False))


def until_text(until):
    return time.strftime("%d.%m.%Y", time.localtime(until)) if until else "—"


# --- лимиты Free ------------------------------------------------------------------------------------
def can_add_pattern(chat_id, name):
    """Free: не больше FREE["patterns"] разных паттернов (примеров одного — сколько угодно)."""
    if is_pro(chat_id):
        return True
    names = {p["name"] for p in storage.get_patterns(chat_id)}
    return name in names or len(names) < FREE["patterns"]


def allowed_patterns(chat_id, patterns, now=None):
    """Free: ищем только первые FREE["patterns"] паттернов (по порядку добавления), остальные ждут Pro."""
    if is_pro(chat_id, now):
        return patterns
    keep = list(dict.fromkeys(p["name"] for p in patterns))[:FREE["patterns"]]
    return [p for p in patterns if p["name"] in keep]


def locked_names(chat_id, now=None):
    """Паттерны, которые на Free не ищутся."""
    if is_pro(chat_id, now):
        return []
    names = list(dict.fromkeys(p["name"] for p in storage.get_patterns(chat_id)))
    return names[FREE["patterns"]:]


def days_left(until, now=None):
    now = now or time.time()
    return max(0, int((until - now + DAY - 1) // DAY)) if until else 0


def use_scan(chat_id, now=None):
    """Засчитать ручной скан. False — у Free на сегодня сканы кончились."""
    if is_pro(chat_id, now):
        return True
    today = time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))

    def change(user):
        usage = user.setdefault("usage", {})
        if usage.get("day") != today:
            usage.update(day=today, scans=0)
        if usage["scans"] >= FREE["scans"]:
            return False
        usage["scans"] += 1
        return True
    return storage.update_user(chat_id, change)


def scans_left(chat_id, now=None):
    today = time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))
    usage = storage.get_user(chat_id).get("usage", {})
    used = usage.get("scans", 0) if usage.get("day") == today else 0
    return max(0, FREE["scans"] - used)


# --- Telegram Stars -----------------------------------------------------------------------------------
_links = {}


def stars_links(api_post):
    """Ссылки на оплату звёздами: {"month": url, "year": url}. api_post(метод, данные) → ответ JSON.
    Ссылки общие для всех: кто заплатил, Telegram сообщает в successful_payment."""
    for plan, (days, stars, _, title) in PLANS.items():
        if plan in _links:
            continue
        data = {"title": title, "payload": f"pro_{plan}_{stars}", "currency": "XTR",
                "description": "Pattern Finder Pro: все паттерны, автоскан, нейросеть, разметка TradingView.",
                "prices": json.dumps([{"label": title, "amount": stars}])}
        if plan == "month":
            data["subscription_period"] = MONTH_SECONDS          # Telegram сам продлевает каждые 30 дней
        answer = api_post("createInvoiceLink", data)
        if answer.get("ok"):
            _links[plan] = answer["result"]
    return dict(_links)


def parse_payload(payload):
    """«pro_month_500» → ("month", 500); старый формат «pro_month» → ("month", None)."""
    parts = (payload or "").split("_")
    if len(parts) < 2 or parts[0] != "pro" or parts[1] not in PLANS:
        return None, None
    price = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    return parts[1], price


def valid_stars(payment):
    """Платёж звёздами честный: наш план, валюта XTR, сумма = цене из счёта (и не ноль)."""
    plan, price = parse_payload(payment.get("invoice_payload"))
    if plan is None or payment.get("currency") != "XTR":
        return False
    amount = payment.get("total_amount") or 0
    return amount > 0 and amount == (price if price is not None else PLANS[plan][1])


def check_pre_checkout(query):
    """Можно ли принять платёж (Telegram спрашивает перед списанием)."""
    return valid_stars(query)


def stars_paid(chat_id, payment, now=None):
    """successful_payment → продлить подписку. Возвращает текст для пользователя (или None — повтор)."""
    now = now or time.time()
    plan, _ = parse_payload(payment.get("invoice_payload"))
    if not valid_stars(payment):
        print(f"⚠️ Подозрительный платёж от {chat_id}: {payment}")
        return None
    charge = payment.get("telegram_payment_charge_id") or str(uuid.uuid4())
    if not storage.add_payment(charge, chat_id, "stars", plan, payment.get("total_amount", 0), "XTR",
                               data=payment):
        return None
    if payment.get("subscription_expiration_date"):              # подписка: Telegram сказал, до какого числа
        until = extend(chat_id, until=payment["subscription_expiration_date"], source="stars", recurring=True,
                       now=now)
    else:
        until = extend(chat_id, PLANS[plan][0], source="stars", now=now)
    renew = " Продление — автоматически каждые 30 дней, отменить можно в настройках Telegram." \
        if payment.get("subscription_expiration_date") else ""
    return f"💎 Спасибо! Pro активен до {until_text(until)}.{renew}"


def stars_refunded(chat_id, refund):
    """refunded_payment: звёзды вернулись пользователю — Pro снимаем. True — было что снимать."""
    charge = refund.get("telegram_payment_charge_id")
    if not charge or not storage.set_payment_status(charge, "refunded"):
        return False
    revoke(chat_id)
    return True


# --- ЮKassa -----------------------------------------------------------------------------------------
YOOKASSA_API = "https://api.yookassa.ru/v3/payments"
PENDING_HOURS = 24                    # неоплаченную ссылку через сутки забываем


def yookassa_on():
    return bool(os.getenv("YOOKASSA_SHOP_ID") and os.getenv("YOOKASSA_SECRET_KEY"))


def _auth():
    return os.getenv("YOOKASSA_SHOP_ID"), os.getenv("YOOKASSA_SECRET_KEY")


def yookassa_create(chat_id, plan, http=requests):
    """Платёж в ЮKassa → ссылка на страницу оплаты (или None, если ЮKassa не настроена / ошибка)."""
    if not yookassa_on() or plan not in PLANS:
        return None
    days, _, rub, title = PLANS[plan]
    body = {"amount": {"value": f"{rub:.2f}", "currency": "RUB"}, "capture": True,
            "confirmation": {"type": "redirect", "return_url": os.getenv("YOOKASSA_RETURN_URL", "https://t.me")},
            "description": f"Pattern Finder: {title}", "metadata": {"chat_id": str(chat_id), "plan": plan}}
    response = http.post(YOOKASSA_API, json=body, auth=_auth(), timeout=20,
                         headers={"Idempotence-Key": str(uuid.uuid4())})
    if response.status_code not in (200, 201):
        print("ЮKassa не создала платёж:", response.status_code, response.text[:200])
        return None
    payment = response.json()
    storage.update_user(chat_id, lambda u: u.setdefault("pending", []).append(
        {"id": payment["id"], "plan": plan, "at": time.time()}))
    return payment["confirmation"]["confirmation_url"]


def yookassa_honest(payment, chat_id, plan):
    """Платёж ЮKassa тот самый: сумма и валюта = цене плана, создан для этого пользователя, не тестовый."""
    amount = payment.get("amount") or {}
    try:
        value = float(amount.get("value", 0))
    except (TypeError, ValueError):
        return False
    meta = payment.get("metadata") or {}
    test_ok = not payment.get("test") or os.getenv("YOOKASSA_ALLOW_TEST") == "1"
    return (amount.get("currency") == "RUB" and abs(value - PLANS[plan][2]) < 0.01
            and str(meta.get("chat_id")) == str(chat_id) and meta.get("plan") == plan and test_ok)


def yookassa_check(chat_id=None, http=requests, now=None):
    """Спросить ЮKassa о неоплаченных ссылках. Возвращает [(chat_id, текст)] — кому сообщить."""
    if not yookassa_on():
        return []
    now = now or time.time()
    news = []
    users = storage.load_users()
    for key, user in users.items():
        if chat_id is not None and key != str(chat_id):
            continue
        for item in list(user.get("pending", [])):
            try:
                response = http.get(f"{YOOKASSA_API}/{item['id']}", auth=_auth(), timeout=20)
                payment = response.json() if response.status_code == 200 else {}
            except requests.RequestException:
                continue
            state = payment.get("status")
            done = state in ("succeeded", "canceled") or now - item["at"] > PENDING_HOURS * 3600
            if state == "succeeded" and payment.get("paid") and not yookassa_honest(payment, key, item["plan"]):
                print(f"⚠️ Подозрительный платёж ЮKassa {item['id']} для {key}")
            elif state == "succeeded" and payment.get("paid"):
                amount = float(payment.get("amount", {}).get("value", 0))
                if storage.add_payment(item["id"], key, "yookassa", item["plan"], amount, "RUB", data=payment):
                    until = extend(int(key), PLANS[item["plan"]][0], source="yookassa", now=now)
                    news.append((int(key), f"💎 Оплата прошла! Pro активен до {until_text(until)}."))
            if done:
                storage.update_user(key, lambda u, pid=item["id"]: u.__setitem__(
                    "pending", [p for p in u.get("pending", []) if p["id"] != pid]))
    return news


# --- напоминания ------------------------------------------------------------------------------------
def reminders(now=None):
    """[(chat_id, текст)]: подписка кончится через ≤3 дня (кроме автопродления звёздами) или кончилась."""
    now = now or time.time()
    out = []
    for key, user in storage.load_users().items():
        sub = user.get("sub") or {}
        until = sub.get("until") or 0
        if not until or is_admin(int(key)) or sub.get("forever"):
            continue
        left = until - now
        window = (1 if sub.get("source") == "trial" else REMIND_DAYS) * DAY     # пробный — за сутки
        if 0 < left <= window and not sub.get("recurring") and sub.get("reminded") != until:
            what = "Пробный Pro" if sub.get("source") == "trial" else "Pro"
            out.append((int(key), f"⏰ {what} закончится {until_text(until)}. Продлить — «💎 Подписка»."))
            storage.update_user(key, lambda u, x=until: _sub(u).__setitem__("reminded", x))
        elif left <= 0 and not sub.get("expired_told"):
            out.append((int(key), "Pro закончился — бот перешёл на Free: 2 паттерна, 3 скана в день, "
                                  "без автоскана и нейросети. Вернуть всё — «💎 Подписка»."))

            def expire(u):
                _sub(u)["expired_told"] = True
                u["autoscan"] = False                    # автоскан — функция Pro
            storage.update_user(key, expire)
    return out


# --- статистика для админа ----------------------------------------------------------------------------
def history(chat_id):
    """Оплаты пользователя (новые сверху)."""
    return [p for p in reversed(storage.payments()) if p["chat_id"] == str(chat_id)]


def stats(now=None):
    now = now or time.time()
    users = storage.load_users()
    subs = [u.get("sub") or {} for u in users.values()]
    pro = sum(bool(s.get("forever")) or ((s.get("until") or 0) > now and s.get("source") != "trial") for s in subs)
    trial = sum((s.get("until") or 0) > now and s.get("source") == "trial" for s in subs)
    paid = storage.payments(since=now - 30 * DAY)
    stars = sum(p["amount"] for p in paid if p["currency"] == "XTR")
    rub = sum(p["amount"] for p in paid if p["currency"] == "RUB")
    auto = sum(bool(u.get("autoscan")) and ((u.get("sub") or {}).get("until") or 0) > now
               for u in users.values())
    return (f"👥 Пользователей: {len(users)}\n💎 Pro (платные): {pro} · пробный: {trial}\n"
            f"🔔 Автоскан включён: {auto}\n💰 За 30 дней: {len(paid)} оплат — ⭐ {stars:.0f}, ₽ {rub:.0f}")
