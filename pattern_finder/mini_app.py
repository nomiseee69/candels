"""Loopback-only Telegram Mini App backend. The bot alone consumes Telegram updates."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from socketserver import ThreadingMixIn
from urllib.parse import parse_qsl, urlsplit
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

import billing
import discounts
import mini_orders
import partner_finance as partners
import partner_payouts
import notifier
import storage
import subscription_guard as guard

ROOT = Path(__file__).with_name("miniapp")
AUTH_TTL = 900


def authenticate(raw, token, now=None):
    fields = parse_qsl(raw, strict_parsing=True, max_num_fields=32)
    values = dict(fields)
    if not token or len(values) != len(fields):
        raise ValueError
    supplied = values.pop("hash", "")
    data = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    secret = hmac.digest(b"WebAppData", token.encode(), hashlib.sha256)
    expected = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
    if not supplied.isascii() or not hmac.compare_digest(expected, supplied):
        raise ValueError
    age = (time.time() if now is None else now) - int(values.get("auth_date", "0"))
    if not -30 <= age <= AUTH_TTL:
        raise ValueError
    user = json.loads(values.get("user", "null"))
    if not isinstance(user, dict) or type(user.get("id")) is not int or not 0 < user["id"] < 2**52:
        raise ValueError
    return user["id"]


PARTNER_ROUTES = {"state", "register", "challenge", "verify", "disconnect", "request"}
TON_SDK = "https://unpkg.com/@tonconnect/ui@3.1.0/dist/tonconnect-ui.min.js"


def manifest_config():
    domain = os.getenv("TON_CONNECT_APP_DOMAIN", "findercandles.duckdns.org")
    if not domain or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-." for c in domain) or "." not in domain:
        raise partners.Error("Домен TON Connect не настроен.")
    url = "https://" + domain
    manifest = url + "/tonconnect-manifest.json"
    if os.getenv("TON_CONNECT_MANIFEST_URL", manifest) != manifest:
        raise partners.Error("Адрес манифеста TON Connect не соответствует домену.")
    return domain, manifest, {"url": url, "name": "Finder Candles Partners", "iconUrl": url + "/tonconnect-icon.png"}


def partner_action(buyer, action, data):
    if action == "state":
        result = partner_payouts.overview(buyer)
        if result["registered"]:
            _, result["manifest"], _ = manifest_config()
            result["sdk"] = TON_SDK
        return result
    if action == "register":
        return {"link": partners.link(buyer)}
    if action == "disconnect":
        partner_payouts.disconnect(buyer)
        return {}
    if not partners.enabled():
        raise partners.Error("Партнёрская программа пока недоступна.")
    if action == "challenge":
        manifest_config()
        return {"payload": partner_payouts.challenge(buyer)}
    if action == "verify":
        domain, _, _ = manifest_config()
        partner_payouts.bind(buyer, data.get("account"), data.get("proof"), domain)
        return {}
    key = partner_payouts.request(buyer, partners.amount(data.get("amount")), billing.admins())
    return {"request": key}


def state(buyer, payload=None):
    status = billing.status(buyer, provision=False)
    offer = discounts.active(buyer)
    percent = offer["percent"] if offer else None
    plans = [{"key": key, "days": days, "base": stars, "amount": discounts.price(stars, percent) if offer else stars,
              "recurring": key == "month" and not offer} for key, (days, stars, _, _) in billing.PLANS.items()]
    with storage.transaction() as db:
        guard.schema(db)
        blocked = bool(guard.paid_until(db, buyer, time.time()))
        charge = guard.recurring_charge(db, buyer) if blocked and status.get("recurring") else None
        cancelled = bool(charge and db.execute("SELECT 1 FROM pro_renewal_cancelled WHERE charge=?", (charge,)).fetchone())
        paid = bool(payload and db.execute("SELECT 1 FROM payments WHERE chat_id=? AND status='succeeded' AND source='stars' "
                    "AND json_extract(data,'$.invoice_payload')=? LIMIT 1", (str(buyer), payload)).fetchone())
    return {"pro": status["pro"], "until": status["until"], "recurring": bool(status.get("recurring") and blocked),
            "cancelled": cancelled, "can_cancel": bool(charge and not cancelled), "purchase_blocked": blocked,
            "scans_left": None if status["pro"] else billing.scans_left(buyer),
            "trial": bool(status["trial"] and status["pro"]), "discount": percent, "plans": plans, "paid": paid}


def cancel_renewal(buyer, api=None):
    status = billing.status(buyer, provision=False)
    with storage.transaction() as db:
        guard.schema(db)
        charge = guard.recurring_charge(db, buyer) if status["pro"] and status.get("recurring") else None
        if not charge:
            raise ValueError("Управляй продлением в настройках Telegram.")
        if db.execute("SELECT 1 FROM pro_renewal_cancelled WHERE charge=?", (charge,)).fetchone():
            return
    response = (api or notifier.api_post)("editUserStarSubscription", {
        "user_id": buyer, "telegram_payment_charge_id": charge, "is_canceled": True})
    if not response.get("ok"):
        raise ValueError("Управляй продлением в настройках Telegram.")
    with storage.transaction() as db:
        db.execute("INSERT OR IGNORE INTO pro_renewal_cancelled VALUES (?)", (charge,))


def invoice(buyer, plan, discounted, api=None):
    if not isinstance(plan, str) or plan not in billing.PLANS or type(discounted) is not bool:
        raise ValueError("Неизвестный тариф.")
    with storage.transaction() as db:
        if guard.paid_until(db, buyer, time.time()):
            raise ValueError(guard.BLOCKED)
    days, amount, _, _ = billing.PLANS[plan]
    title = f"Finder Candles Pro · {days} дней"
    if discounted:
        order = discounts.invoice(buyer, plan)
        if not order:
            raise ValueError("Скидка изменилась. Обнови экран.")
        data = {key: value for key, value in order.items() if key not in ("chat_id", "start_parameter", "provider_token")}
    else:
        if discounts.active(buyer):
            raise ValueError("Доступна персональная скидка. Обнови экран.")
        data = {"title": title, "description": "Поиск паттернов, автоскан и разметка TradingView.",
                "payload": mini_orders.create(buyer, plan, amount), "currency": "XTR",
                "prices": json.dumps([{"label": title, "amount": amount}])}
        if plan == "month":
            data["subscription_period"] = billing.MONTH_SECONDS
    reply = (api or notifier.api_post)("createInvoiceLink", data)
    url = reply.get("result") if reply.get("ok") else None
    if not isinstance(url, str):
        raise RuntimeError
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "t.me":
        raise RuntimeError
    return {"url": url, "payload": data["payload"]}


TYPES = {"/": ("index.html", "text/html"), "/app.css": ("app.css", "text/css"),
         "/app.js": ("app.js", "text/javascript"),
         "/tonconnect-icon.png": ("tonconnect-icon.png", "image/png")}
HEADERS = [("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
           ("Referrer-Policy", "no-referrer"),
           ("Content-Security-Policy", "default-src 'none'; script-src 'self' https://telegram.org https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self' https:; img-src 'self' https: data:; base-uri 'none'; form-action 'none'")]


def application(env, start):
    def response(status, body, mime="application/json", extra=()):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        start(status, HEADERS + list(extra) + [("Content-Type", mime + ("; charset=utf-8" if not mime.startswith("image/") else "")),
                                                  ("Content-Length", str(len(data)))])
        return [data]

    path, method = env["PATH_INFO"], env["REQUEST_METHOD"]
    if method == "GET" and path in TYPES:
        filename, mime = TYPES[path]
        return response("200 OK", (ROOT / filename).read_bytes(), mime)
    if method == "GET" and path == "/health":
        return response("200 OK", {"ok": True})
    if method == "GET" and path == "/tonconnect-manifest.json":
        try:
            _, _, manifest = manifest_config()
        except partners.Error:
            return response("503 Service Unavailable", {"error": "TON Connect недоступен."})
        return response("200 OK", manifest, extra=(("Access-Control-Allow-Origin", "*"),))
    affiliate = path.removeprefix("/api/partners/") if path.startswith("/api/partners/") else None
    if method != "POST" or (path not in ("/api/state", "/api/invoice", "/api/cancel") and affiliate not in PARTNER_ROUTES):
        return response("404 Not Found", {"error": "Страница недоступна."})
    raw = env.get("HTTP_X_TELEGRAM_INIT_DATA", "")
    try:
        buyer = authenticate(raw, notifier.TOKEN)
    except (ValueError, TypeError):
        return response("401 Unauthorized", {"error": "Открой приложение заново из Telegram."})
    try:
        length = int(env.get("CONTENT_LENGTH") or "0")
        if not 0 < length <= (24000 if affiliate == "verify" else 2048):
            raise ValueError("Некорректный запрос.")
        data = json.loads(env["wsgi.input"].read(length))
        if not isinstance(data, dict):
            raise ValueError("Некорректный запрос.")
        if affiliate in PARTNER_ROUTES:
            return response("200 OK", partner_action(buyer, affiliate, data))
        if path == "/api/state":
            payload = data.get("payload")
            return response("200 OK", state(buyer, payload if isinstance(payload, str) else None))
        if path == "/api/cancel":
            cancel_renewal(buyer)
            return response("200 OK", state(buyer))
        return response("200 OK", invoice(buyer, data.get("plan"), data.get("discounted")))
    except partners.Error as error:
        return response("400 Bad Request", {"error": str(error)})
    except ValueError as error:
        # JSON parsing errors can contain input details, so expose only our own text.
        message = str(error) if str(error) in (
            guard.BLOCKED, "Управляй продлением в настройках Telegram.",
            "Некорректный запрос.", "Неизвестный тариф.", "Скидка изменилась. Обнови экран.",
            "Доступна персональная скидка. Обнови экран.", "Слишком много открытых счетов. Повтори позже.") else "Некорректный запрос."
        return response("400 Bad Request", {"error": message})
    except Exception:
        return response("503 Service Unavailable", {"error": "Сервис временно недоступен. Попробуй позже."})


class Server(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class Handler(WSGIRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if not notifier.TOKEN:
        raise SystemExit("TELEGRAM_TOKEN is required")
    if not mini_orders.app_url():
        raise SystemExit("An HTTPS MINI_APP_URL is required")
    with make_server("127.0.0.1", 8765, application, Server, Handler) as server:
        server.serve_forever()
