"""Данные с Binance: свечи и список самых торгуемых монет."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

API = "https://api.binance.com/api/v3"
RETRIES = 3
MAX_LIMIT = 1000                # больше свечей за один запрос Binance не отдаёт
WORKERS = 8                     # сколько монет качаем одновременно
TOP_CACHE_SECONDS = 6 * 3600    # список топ-монет обновляем раз в 6 часов

# монеты, которые всегда стоят ~1$ (или привязаны к золоту/другой монете): паттернов на них нет
PEGGED = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USD1", "PYUSD", "USDE", "USDS", "RLUSD",
    "BFUSD", "XUSD", "AEUR", "EUR", "EURI", "GBP", "PAXG", "XAUT", "WBTC", "WBETH", "BETH",
}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")

# если биржа недоступна — ищем хотя бы по этим
FALLBACK_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


def _get(path, params):
    """GET к Binance с повторами: при обрыве связи или лимите запросов ждём и пробуем снова.
    Возвращает JSON или None."""
    for attempt in range(RETRIES):
        try:
            response = requests.get(f"{API}/{path}", params=params, timeout=10)
        except requests.RequestException as error:
            print(f"Binance: ошибка сети ({error.__class__.__name__}), попытка {attempt + 1}")
            time.sleep(1 + attempt)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code in (418, 429):             # слишком много запросов
            wait = int(response.headers.get("Retry-After", 5))
            print(f"Binance просит подождать {wait} с")
            time.sleep(wait)
            continue
        print(f"Binance: ошибка {response.status_code} для {params}")
        return None                                        # например, монеты нет — повторять незачем
    return None


def get_candles(symbol, interval, limit):
    """Закрытые свечи (последняя, ещё не закрытая, отбрасывается)."""
    # +1 — последняя свеча ещё не закрыта и будет отброшена; Binance отдаёт максимум 1000
    data = _get("klines", {"symbol": symbol, "interval": interval, "limit": min(limit + 1, MAX_LIMIT)})
    if not data:
        return None
    return data[:-1]


def fetch_many(symbols, interval, limit, fetch=get_candles, workers=WORKERS):
    """Свечи сразу для многих монет, параллельно. Возвращает {монета: свечи}, порядок как в symbols."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: fetch(s, interval, limit), symbols))
    return dict(zip(symbols, results))


# --- топ монет по объёму ----------------------------------------------------------------------
def pick_top(tickers, n, now_ms):
    """Из тикеров Binance выбирает n USDT-пар с наибольшим оборотом за сутки.
    Отсекает стейблкоины, токены с плечом и пары, которые больше не торгуются."""
    rows = []
    for t in tickers:
        symbol = t["symbol"]
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if not base or base in PEGGED or base.endswith(LEVERAGED):
            continue
        if now_ms - int(t.get("closeTime", 0)) > 2 * 86400 * 1000:   # торги остановлены
            continue
        volume = float(t.get("quoteVolume", 0))
        if volume > 0:
            rows.append((volume, symbol))
    rows.sort(reverse=True)
    return [symbol for _, symbol in rows[:n]]


_top_cache = {"at": 0.0, "symbols": []}
_top_lock = threading.Lock()


def top_symbols(n, now=None):
    """n самых торгуемых монет. Список кэшируется на 6 часов; если биржа не ответила —
    берём прошлый список или запасные 10 монет."""
    now = now or time.time()
    with _top_lock:
        if now - _top_cache["at"] > TOP_CACHE_SECONDS or len(_top_cache["symbols"]) < n:
            tickers = _get("ticker/24hr", {"type": "MINI"})
            if tickers:
                _top_cache["symbols"] = pick_top(tickers, 200, now * 1000)
                _top_cache["at"] = now
        symbols = _top_cache["symbols"] or FALLBACK_SYMBOLS
    return symbols[:n]


def candles_to_df(candles):
    df = pd.DataFrame(candles)
    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    price_cols = ["open", "high", "low", "close", "volume"]
    df[price_cols] = df[price_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["is_green"] = df["close"] > df["open"]
    return df
