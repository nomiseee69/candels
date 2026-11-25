"""Loopback-only Telegram Mini App backend. The bot alone consumes Telegram updates."""

import hashlib
import hmac
import json
from pathlib import Path
import time
from socketserver import ThreadingMixIn
from urllib.parse import parse_qsl, urlsplit
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

import billing
import discounts
import mini_orders
import notifier
import storage

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


def state(buyer, payload=None):
    status = billing.status(buyer, provision=False)
    offer = discounts.active(buyer)
    percent = offer["percent"] if offer else None
    plans = [{"key": key, "days": days, "base": stars, "amount": discounts.price(stars, percent) if offer else stars,
              "recurring": key == "month" and not offer} for key, (days, stars, _, _) in billing.PLANS.items()]
    paid = False
    if payload:
        with storage.transaction() as db:
            paid = db.execute("SELECT 1 FROM payments WHERE chat_id=? AND status='succeeded' AND source='stars' "
                              "AND json_extract(data,'$.invoice_payload')=? LIMIT 1", (str(buyer), payload)).fetchone() is not None
    return {"pro": status["pro"], "until": status["until"], "recurring": bool(status.get("recurring") and status["pro"]),
            "trial": bool(status["trial"] and status["pro"]), "discount": percent, "plans": plans, "paid": paid}


def invoice(buyer, plan, discounted, api=None):
    if not isinstance(plan, str) or plan not in billing.PLANS or type(discounted) is not bool:
        raise ValueError("Неизвестный тариф.")
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
         "/app.js": ("app.js", "text/javascript")}
HEADERS = [("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
           ("Referrer-Policy", "no-referrer"),
           ("Content-Security-Policy", "default-src 'none'; script-src 'self' https://telegram.org; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")]


def application(env, start):
    def response(status, body, mime="application/json"):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        start(status, HEADERS + [("Content-Type", mime + "; charset=utf-8"), ("Content-Length", str(len(data)))])
        return [data]

    path, method = env["PATH_INFO"], env["REQUEST_METHOD"]
    if method == "GET" and path in TYPES:
        filename, mime = TYPES[path]
        return response("200 OK", (ROOT / filename).read_bytes(), mime)
    if method != "POST" or path not in ("/api/state", "/api/invoice"):
        return response("404 Not Found", {"error": "Страница недоступна."})
    raw = env.get("HTTP_X_TELEGRAM_INIT_DATA", "")
    try:
        buyer = authenticate(raw, notifier.TOKEN)
    except (ValueError, TypeError):
        return response("401 Unauthorized", {"error": "Открой приложение заново из Telegram."})
    try:
        length = int(env.get("CONTENT_LENGTH") or "0")
        if not 0 < length <= 2048:
            raise ValueError("Некорректный запрос.")
        data = json.loads(env["wsgi.input"].read(length))
        if not isinstance(data, dict):
            raise ValueError("Некорректный запрос.")
        if path == "/api/state":
            payload = data.get("payload")
            return response("200 OK", state(buyer, payload if isinstance(payload, str) else None))
        return response("200 OK", invoice(buyer, data.get("plan"), data.get("discounted")))
    except ValueError as error:
        # JSON parsing errors can contain input details, so expose only our own text.
        message = str(error) if str(error) in (
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
