"""Подписка, оплата, лимиты Free, база SQLite, резервные копии, админ."""

import json
import os
import time

import billing
import storage

DAY = 86400


def use_db(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    return original


class FakeHttp:
    """Вместо ЮKassa: создаёт платёж и отвечает заданным статусом."""

    def __init__(self, status="pending"):
        self.status, self.created = status, []

    class Response:
        def __init__(self, code, data):
            self.status_code, self._data, self.text = code, data, json.dumps(data)

        def json(self):
            return self._data

    def post(self, url, json=None, auth=None, timeout=None, headers=None):
        self.created.append(json)
        return self.Response(200, {"id": f"pay{len(self.created)}",
                                   "confirmation": {"confirmation_url": "https://yoomoney.ru/checkout/x"}})

    def get(self, url, auth=None, timeout=None):
        return self.Response(200, {"id": url.rsplit("/", 1)[1], "status": self.status, "paid": self.status == "succeeded",
                                   "amount": {"value": "990.00", "currency": "RUB"}})


def test_json_moves_into_database_and_backups(tmp_path):
    original = use_db(tmp_path)
    try:
        with open(storage.FILE, "w", encoding="utf-8") as f:
            json.dump({"5": {"patterns": [{"id": "1", "name": "FVG", "photo": None}]}}, f)
        assert storage.get_patterns(5)[0]["name"] == "FVG"                 # переехало в базу
        assert os.path.exists(storage.db_path()) and os.path.exists(storage.FILE + ".migrated")
        storage.update_user(6, lambda u: u.update(autoscan=True))
        assert set(storage.load_users()) == {"5", "6"}
        storage.BACKUP_DIR = str(tmp_path / "backups")
        for day in range(9):
            storage.backup(now=time.time() - day * DAY)
        assert len(os.listdir(storage.BACKUP_DIR)) == storage.BACKUP_KEEP
    finally:
        storage.FILE = original
        storage.BACKUP_DIR = "backups"


def test_trial_then_free_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_IDS", "1")
    original = use_db(tmp_path)
    try:
        now = time.time()
        assert billing.is_pro(5, now)                                       # новичку — пробный Pro
        later = now + (billing.TRIAL_DAYS + 1) * DAY
        assert not billing.is_pro(5, later)
        storage.add_pattern(5, "FVG", None, builtin=True)
        storage.add_pattern(5, "BOS", None, builtin=True)
        storage.update_user(5, lambda u: u["sub"].update(until=now - 1))   # пробный кончился
        assert billing.can_add_pattern(5, "FVG") and not billing.can_add_pattern(5, "Канал")
        assert [billing.use_scan(5, now) for _ in range(4)] == [True, True, True, False]
        assert billing.scans_left(5, now) == 0 and billing.use_scan(5, now + DAY)   # новый день — снова можно
        assert billing.is_pro(1) and billing.can_add_pattern(1, "Канал")   # админ — всегда Pro
    finally:
        storage.FILE = original


def test_stars_subscription(tmp_path):
    original = use_db(tmp_path)
    billing._links.clear()
    try:
        asked = []
        links = billing.stars_links(lambda method, data: asked.append(data) or {"ok": True, "result": f"https://t.me/$x{len(asked)}"})
        assert set(links) == {"month", "year"}
        month = next(d for d in asked if d["payload"] == "pro_month")
        assert month["currency"] == "XTR" and month["subscription_period"] == 30 * DAY
        assert "subscription_period" not in next(d for d in asked if d["payload"] == "pro_year")
        assert billing.check_pre_checkout({"invoice_payload": "pro_month", "currency": "XTR"})
        assert not billing.check_pre_checkout({"invoice_payload": "pro_month", "currency": "USD"})
        now = time.time()
        storage.update_user(7, lambda u: u.update(sub={"trial_used": True}))
        paid = {"invoice_payload": "pro_month", "total_amount": 500, "currency": "XTR",
                "telegram_payment_charge_id": "c1", "subscription_expiration_date": int(now + 30 * DAY)}
        assert "Pro активен" in billing.stars_paid(7, paid, now)
        assert billing.stars_paid(7, paid, now) is None                    # тот же платёж второй раз — не считаем
        st = billing.status(7, now)
        assert st["pro"] and st["recurring"] and abs(st["until"] - (now + 30 * DAY)) < 2
        year = {"invoice_payload": "pro_year", "total_amount": 4800, "currency": "XTR", "telegram_payment_charge_id": "c2"}
        billing.stars_paid(7, year, now)
        assert billing.status(7, now)["until"] > now + 390 * DAY            # год добавился к месяцу
    finally:
        storage.FILE = original
        billing._links.clear()


def test_yookassa_payment(tmp_path, monkeypatch):
    monkeypatch.setenv("YOOKASSA_SHOP_ID", "1")
    monkeypatch.setenv("YOOKASSA_SECRET_KEY", "k")
    original = use_db(tmp_path)
    try:
        http = FakeHttp("pending")
        url = billing.yookassa_create(8, "month", http)
        assert url.startswith("https://") and http.created[0]["amount"] == {"value": "990.00", "currency": "RUB"}
        assert billing.yookassa_check(http=http) == []                      # ещё не оплачено — ждём
        http.status = "succeeded"
        news = billing.yookassa_check(http=http)
        assert news and news[0][0] == 8 and "Оплата прошла" in news[0][1]
        assert billing.yookassa_check(http=http) == []                      # ссылка закрыта, второй раз не продлеваем
        assert billing.is_pro(8) and storage.payments()[0]["currency"] == "RUB"
    finally:
        storage.FILE = original


def test_reminders_once(tmp_path):
    original = use_db(tmp_path)
    try:
        now = time.time()
        billing.status(9, now)                                              # пробный на 7 дней
        assert billing.reminders(now) == []
        soon = now + (billing.TRIAL_DAYS - 2) * DAY
        assert len(billing.reminders(soon)) == 1 and billing.reminders(soon) == []
        gone = now + (billing.TRIAL_DAYS + 1) * DAY
        assert "Free" in billing.reminders(gone)[0][1] and billing.reminders(gone) == []
    finally:
        storage.FILE = original


def test_bot_payments_paywall_and_admin(tmp_path, monkeypatch):
    import bot
    from test_menus import Recorder
    monkeypatch.setenv("ADMIN_IDS", "1")
    rec = Recorder()
    original = use_db(tmp_path)
    billing._links.clear()
    saved = billing.stars_links
    billing.stars_links = lambda api: {"month": "https://t.me/$m", "year": "https://t.me/$y"}
    try:
        now = time.time()
        storage.update_user(5, lambda u: u.update(sub={"trial_used": True, "until": now - 1}))
        bot.handle_callback({"id": "q", "data": "auto", "message": {"chat": {"id": 5}, "message_id": 3}})
        assert "Автоскан — функция Pro" in rec.sent[-1][0] and not storage.get_autoscan(5)
        bot.handle_message({"chat": {"id": 5}, "text": "/subscribe"})
        text, keyboard = rec.sent[-1]
        urls = [b.get("url") for row in keyboard["inline_keyboard"] for b in row]
        assert "https://t.me/$m" in urls and "не финансовый совет" in text
        bot.handle_message({"chat": {"id": 5}, "successful_payment": {
            "invoice_payload": "pro_year", "total_amount": 4800, "currency": "XTR", "telegram_payment_charge_id": "z"}})
        assert "Pro активен" in rec.sent[-2][0] or "Pro активен" in rec.sent[-1][0]
        assert billing.is_pro(5)
        bot.handle_message({"chat": {"id": 1}, "text": "/grant 6 30"})
        assert billing.is_pro(6) and any("Pro до" in t for t, _ in rec.sent)
        bot.handle_message({"chat": {"id": 5}, "text": "/grant 7 30"})     # не админ — команда не работает
        assert not storage.get_user(7).get("sub")
        bot.handle_message({"chat": {"id": 1}, "text": "/admin"})
        assert "Пользователей" in rec.sent[-1][0]
    finally:
        rec.restore()
        storage.FILE = original
        billing.stars_links = saved


def test_brain_rate_limit():
    import bot
    bot.brain_calls.clear()
    now = time.time()
    assert all(bot.brain_allowed(11, now) for _ in range(bot.BRAIN_PER_HOUR))
    assert not bot.brain_allowed(11, now) and bot.brain_allowed(11, now + 3601)


def test_free_profile_and_old_patterns(tmp_path, monkeypatch):
    """После пробного: в меню и профиле видно Free и сколько осталось; старые паттерны сверх
    лимита не ищутся; автоскан выключается и закрыт замком."""
    import bot
    from test_menus import Recorder
    monkeypatch.setenv("ADMIN_IDS", "1")
    rec = Recorder()
    original = use_db(tmp_path)
    try:
        now = time.time()
        for name in ("FVG", "BOS", "Канал"):
            storage.add_pattern(5, name, None, builtin=True)
        storage.update_user(5, lambda u: u.update(autoscan=True))
        text, _ = bot.profile_view(5)
        assert "пробный период" in text and "осталось 7 дн." in text and "После окончания: тариф Free" in text
        storage.update_user(5, lambda u: u["sub"].update(until=now - 1))
        billing.reminders(now)                                         # «Pro закончился» — автоскан выключен
        assert not storage.get_autoscan(5)
        assert [p["name"] for p in bot.search_patterns(5)] == ["FVG", "BOS"]
        text, _ = bot.profile_view(5)
        assert "Тариф: Free" in text and "Канал" in text and "Сканов сегодня осталось: 3 из 3" in text
        assert "Тариф: Free" in bot.menu_text(5)
        names = [b["text"] for row in bot.menu_keyboard(5)["inline_keyboard"] for b in row]
        assert "🔒 Автоскан (Pro)" in names
        billing.extend(5, 30, source="stars")
        assert "осталось 30 дн." in bot.plan_line(5) and len(bot.search_patterns(5)) == 3
    finally:
        rec.restore()
        storage.FILE = original


def test_admin_refund(tmp_path, monkeypatch):
    import bot
    from test_menus import Recorder
    monkeypatch.setenv("ADMIN_IDS", "1")
    rec = Recorder()
    original = use_db(tmp_path)
    calls = []
    saved = bot.api_post
    bot.api_post = lambda method, data: calls.append((method, data)) or {"ok": True, "result": {"amount": 500}}
    try:
        billing.stars_paid(5, {"invoice_payload": "pro_month", "total_amount": 500, "currency": "XTR",
                               "telegram_payment_charge_id": "ch1"})
        assert billing.is_pro(5)
        bot.handle_message({"chat": {"id": 1}, "text": "/refund 5"})
        assert ("refundStarPayment", {"user_id": 5, "telegram_payment_charge_id": "ch1"}) in calls
        assert not billing.is_pro(5) and "Вернул 500" in rec.sent[-1][0]
        bot.handle_message({"chat": {"id": 1}, "text": "/admin"})
        assert "На балансе бота: 500" in rec.sent[-1][0]
    finally:
        bot.api_post = saved
        rec.restore()
        storage.FILE = original
