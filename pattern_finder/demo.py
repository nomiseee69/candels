from shape import photo_to_shape
from market import get_candles, candles_to_df
from matcher import find_best_match_multi
from chart import plot_match
from notifier import send_photo, tradingview_link
PATTERN_NAME = "Двойное дно"
PHOTO = "w_shape.png"
INTERVAL = "4h"
WINDOW = [20,30,40,60,80,120]
RECENT = None
MIN_SCORE = 0.8
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]

shape = photo_to_shape(PHOTO)
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

if not good:
    print(f"\nСвежих совпадений выше {MIN_SCORE:.0%} нет")
else:
    print("\nНайдены совпадения:")
    for score, symbol, start, window_len, df in good[:3]:
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
        send_photo(path, caption)