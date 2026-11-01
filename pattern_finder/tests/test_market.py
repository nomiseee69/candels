"""Больше монет: топ по объёму, загрузка свечей с повторами, много находок, фоновые потоки."""

import threading

import pandas as pd
import requests

import market
import scanner
import storage

NOW_MS = 1_800_000_000_000


def ticker(symbol, volume, close_ms=NOW_MS):
    return {"symbol": symbol, "quoteVolume": str(volume), "closeTime": close_ms}


# --- топ монет ---------------------------------------------------------------------------------
def test_pick_top_filters_and_sorts():
    tickers = [
        ticker("BTCUSDT", 900), ticker("USDCUSDT", 5000), ticker("ETHUSDT", 800),
        ticker("BTCUPUSDT", 700), ticker("SOLBTC", 10_000), ticker("OLDUSDT", 600, close_ms=0),
        ticker("PEPEUSDT", 500), ticker("ZEROUSDT", 0), ticker("FDUSDUSDT", 4000),
    ]
    assert market.pick_top(tickers, 10, NOW_MS) == ["BTCUSDT", "ETHUSDT", "PEPEUSDT"]
    assert market.pick_top(tickers, 2, NOW_MS) == ["BTCUSDT", "ETHUSDT"]


def test_top_symbols_cached_and_fallback():
    calls = []
    original_get, original_cache = market._get, dict(market._top_cache)
    try:
        market._top_cache.update(at=0.0, symbols=[])
        market._get = lambda path, params: calls.append(path) or None        # биржа не отвечает
        assert market.top_symbols(5, now=1000) == market.FALLBACK_SYMBOLS[:5]

        market._get = lambda path, params: calls.append(path) or [
            ticker(f"C{i}USDT", 1000 - i, close_ms=1000 * 1000) for i in range(300)]
        assert market.top_symbols(3, now=1000) == ["C0USDT", "C1USDT", "C2USDT"]
        assert len(market.top_symbols(100, now=2000)) == 100
        assert calls.count("ticker/24hr") == 2               # второй раз — из кэша
        market.top_symbols(3, now=1000 + market.TOP_CACHE_SECONDS + 1)
        assert calls.count("ticker/24hr") == 3               # через 6 часов — обновили
    finally:
        market._get = original_get
        market._top_cache.clear()
        market._top_cache.update(original_cache)


# --- свечи ---------------------------------------------------------------------------------------
def test_get_candles_drops_unclosed_candle():
    original = market._get
    try:
        seen = {}
        market._get = lambda path, params: seen.update(params) or [[i] for i in range(params["limit"])]
        candles = market.get_candles("BTCUSDT", "4h", 300)
        assert seen["limit"] == 301 and len(candles) == 300 and candles[-1] == [299]
    finally:
        market._get = original


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.payload = payload
        self.headers = {"Retry-After": "0"}

    def json(self):
        return self.payload


def test_get_retries_after_network_error_and_rate_limit():
    answers = [requests.ConnectionError("обрыв"), FakeResponse(429), FakeResponse(200, [1, 2])]

    def fake_get(url, params, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    original = market.requests.get, market.time.sleep
    market.requests.get, market.time.sleep = fake_get, lambda s: None
    try:
        assert market._get("klines", {}) == [1, 2]
        answers[:] = [FakeResponse(400)]
        assert market._get("klines", {}) is None             # неверная монета — без повторов
    finally:
        market.requests.get, market.time.sleep = original


def test_fetch_many_keeps_order():

    def fetch(symbol, interval, limit):
        return f"{symbol}-{interval}-{limit}"

    symbols = [f"C{i}USDT" for i in range(40)]
    result = market.fetch_many(symbols, "1h", 10, fetch)
    assert list(result) == symbols
    assert result["C7USDT"] == "C7USDT-1h-10"


# --- сканирование многих монет -----------------------------------------------------------------
def raw_candles(n):
    start = 1_700_000_000_000
    return [[start + i * 3_600_000, "1", "2", "0.5", "1.5", "10"] for i in range(n)]


def test_scan_returns_all_cards_and_skips_young_coins():
    original = scanner.build_detectors
    try:
        scanner.build_detectors = lambda patterns, settings=None: (
            [("FVG", lambda df: {"score": 0.9, "start": len(df) - 5, "window": 5})], [])
        symbols = [f"C{i}USDT" for i in range(30)]
        fetch = lambda s, i, n: raw_candles(50) if s == "C3USDT" else raw_candles(n)
        cards, _ = scanner.scan_user([{"name": "FVG"}], ["1h"], fetch, symbols)
        assert len(cards) == 29                              # не обрезаем до 8, C3 — мало свечей
        assert "C3USDT" not in {c["symbol"] for c in cards}
    finally:
        scanner.build_detectors = original


def test_cards_overview_text():
    df = pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")})
    frame = lambda s, tf: {"symbol": s, "interval": tf, "df": df, "history": False, "caption": "",
                           "items": [{"name": "FVG", "score": 0.91, "kind": "bullish_fvg"},
                                     {"name": "Треугольник", "score": 0.9, "touches": 6}]}
    coins = scanner.group_by_coin([frame(f"C{i}USDT", "4h") for i in range(45)] + [frame("C0USDT", "1h")])
    text = scanner.cards_overview(coins, limit=40)
    assert text.startswith("📋 Ещё 45 монет с находками")
    assert "• C0USDT — 1H: FVG, Треугольник (6 касаний) · 4H: FVG, Треугольник (6 касаний)" in text
    assert text.endswith("…и ещё 5")
    assert scanner.cards_overview([]) == ""


# --- настройки и потоки ---------------------------------------------------------------------------
def test_coins_setting(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    try:
        assert storage.get_coins(1) == 50
        storage.set_coins(1, 100)
        assert storage.get_coins(1) == 100
    finally:
        storage.FILE = original


def test_storage_safe_from_many_threads(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    try:
        def worker(k):
            for j in range(5):
                storage.add_pattern(1, f"P{k}-{j}", "x.jpg")
                storage.add_sent(1, [{"at": 0, "k": k}], now=0)
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(storage.get_patterns(1)) == 40            # ни одно изменение не потерялось
        assert len(storage.get_sent(1)) == 40
    finally:
        storage.FILE = original


def test_bot_card_ids_and_overview():
    import bot
    sent_cards, messages = [], []
    original = bot.send_card, bot.send_message
    bot.send_card = lambda chat_id, card_id, coin: sent_cards.append(card_id)
    bot.send_message = lambda text, chat_id, keyboard=None: messages.append((text, keyboard))
    try:
        df = pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")})
        cards = [{"symbol": f"C{i}USDT", "interval": "1h", "df": df, "history": False, "caption": "",
                  "items": [{"name": "FVG", "score": 0.9 - i / 100, "kind": "bullish_fvg"}]} for i in range(20)]
        coins = scanner.group_by_coin(cards)
        bot.send_cards(5, coins)
        assert len(sent_cards) == bot.MAX_CARDS
        text, keyboard = messages[-1]
        buttons = [b for row in keyboard["inline_keyboard"] for b in row]
        assert len(buttons) == 20 - bot.MAX_CARDS and buttons[0]["text"] == "C8"
        card_id = int(buttons[0]["callback_data"].split(":")[1])
        assert bot.find_card(5, card_id)["symbol"] == "C8USDT"
        bot.send_cards(5, coins[:1])                          # новое сканирование не ломает старые кнопки
        assert bot.find_card(5, card_id)["symbol"] == "C8USDT"
        assert bot.find_card(6, card_id) is None              # чужие находки не видны
    finally:
        bot.send_card, bot.send_message = original
