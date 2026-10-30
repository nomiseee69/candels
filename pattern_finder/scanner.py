from shape import photo_to_shape
from market import get_candles, candles_to_df
from matcher import find_best_match_multi
from chart import plot_match
from notifier import send_photo, tradingview_link
from catalog import SHAPE_PATTERNS
PATTERN_NAME = "Двойное дно"
INTERVAL = "4h"
WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = None
MIN_SCORE = 0.8
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


def scan_market(photo_path):
    shape = photo_to_shape(photo_path)
    results = []
    for symbol in SYMBOLS:
        candles = get_candles(symbol, INTERVAL, 300)
        if not candles:
            continue
        df = candles_to_df(candles)
        start, score, window_len= find_best_match_multi(df["close"].tolist(), shape, WINDOWS, RECENT)
        results.append((score, symbol, start, window_len, df))
        print(f"{symbol}: {score:.0%}, окно {window_len} свечей, начало {df.loc[start, 'open_time']:%Y-%m-%d %H:%M} UTC")

    results.sort(key=lambda r: r[0], reverse=True)
    good = []
    for r in results:
        if r[0] >= MIN_SCORE:
            good.append(r)
    return good[:3]


def send_results(good, chat_id):
    for score, symbol, start, window_len, df in good:
            print(f"  {symbol}: {score:.0%}, окно {window_len}")
            path = f"match_{symbol}.png"
            plot_match(df, symbol, start, window_len, score, path)
        
            begin = df.loc[start, "open_time"]
            end = df.loc[start + window_len - 1, "open_time"]
            caption = (
                    f"🔔 {PATTERN_NAME} на {symbol} {INTERVAL}\n"
                    f"Совпадение: {score:.0%}\n"
                    f"Период: {begin:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC\n"
                    f"{tradingview_link(symbol, INTERVAL)}"
                )
            send_photo(path, caption, chat_id)

def find_shape(df, shape):
    start, score, window_len = find_best_match_multi(df["close"].tolist(), shape, WINDOWS, RECENT)
    if score < MIN_SCORE:
        return None
    return {"score": score, "start": start, "window": window_len}


def scan_user(patterns):
    shapes = []
    skipped = []
    for p in patterns:
        if p["name"] in SHAPE_PATTERNS:
            shapes.append((p["name"], photo_to_shape(p["photo"])))
        else:
            skipped.append(p["name"])

    findings = {}
    if not shapes:
        return findings, skipped

    for symbol in SYMBOLS:
        candles = get_candles(symbol, INTERVAL, 300)
        if not candles:
            continue
        df = candles_to_df(candles)
        items = []
        for name, shape in shapes:
            found = find_shape(df, shape)                  # проверь этот паттерн на этой монете
            if found is not None:                        # нашёлся?
                found["name"] = name
                items.append(found)
        if items:
            findings[symbol] = {"df": df, "items": items}
    return findings, skipped


def send_scan_results(findings, chat_id):
    ordered = sorted(findings.items(), key=lambda kv: len(kv[1]["items"]), reverse=True)
    for symbol, data in ordered[:5]:
        df = data["df"]
        items = data["items"]
        best = max(items, key=lambda it: it["score"])
        path = f"scan_{chat_id}_{symbol}.png"
        plot_match(df, symbol, best["start"], best["window"], best["score"], path)

        names = " + ".join(f"{it['name']} ({it['score']:.0%})" for it in items)
        title = "🔥 Комбинация" if len(items) > 1 else "🔔 Паттерн"
        caption = (
            f"{title} на {symbol} {INTERVAL}\n"
            f"{names}\n"
            f"{tradingview_link(symbol, INTERVAL)}"
        )
        send_photo(path, caption, chat_id)