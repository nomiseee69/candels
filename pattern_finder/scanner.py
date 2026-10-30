from shape import photo_to_shape
from market import get_candles, candles_to_df
from matcher import find_best_match_multi
from chart import plot_match
from notifier import send_photo, tradingview_link

PATTERN_NAME = "Двойное дно"
INTERVAL = "4h"
WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = 3
MIN_SCORE = 0.85
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
        start, score, window_len= find_best_match_multi(df["close"].tolist(), shape, WINDOW, RECENT)
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