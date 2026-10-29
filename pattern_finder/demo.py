from shape import photo_to_shape
from market import get_candles, candles_to_df
from matcher import find_best_match_multi
from chart import plot_match

PHOTO = "w_shape.png"
INTERVAL = "4h"
WINDOW = [20,30,40,60,80]
RECENT = 3
MIN_SCORE = 0.85
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
        plot_match(df, symbol, start, window_len, score, f"match_{symbol}.png")