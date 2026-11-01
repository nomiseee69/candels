"""Smart Money Concepts: детекторы по правилам на свечах.

FVG (Fair Value Gap, имбаланс) — «дыра» в цене из трёх свечей:
  бычий:  low третьей свечи выше high первой — между ними цена не торговалась;
  медвежий: high третьей свечи ниже low первой.
Цена часто возвращается «закрыть» такую дыру, поэтому трейдеры следят
за свежими незакрытыми FVG.
"""

import numpy as np
import pandas as pd

ATR_PERIOD = 14
MIN_GAP_ATR = 0.5          # дыра меньше половины среднего хода свечи — шум
LOOKBACK = 30              # FVG свежий: образовался за последние 30 свечей
MAX_FILLED = 0.99          # заполнен на 100% — уже неинтересен

FVG_NAMES = {"bullish_fvg": "бычий FVG", "bearish_fvg": "медвежий FVG"}


def atr(df, period=ATR_PERIOD):
    """Средний истинный диапазон свечи — «обычный» размер движения на этой монете."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def find_all_fvg(df, lookback=LOOKBACK):
    """Все незакрытые FVG за последние lookback свечей, от свежих к старым."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = atr(df).to_numpy()
    n = len(df)
    found = []

    for i in range(max(2, n - lookback), n):
        first, third = i - 2, i
        size_atr = atrs[i - 1]
        if highs[first] < lows[third]:
            kind, bottom, top = "bullish_fvg", highs[first], lows[third]
        elif lows[first] > highs[third]:
            kind, bottom, top = "bearish_fvg", highs[third], lows[first]
        else:
            continue

        gap = top - bottom
        if size_atr <= 0 or gap < MIN_GAP_ATR * size_atr:
            continue

        # насколько цена после образования уже залезла в дыру
        after = slice(third + 1, n)
        if kind == "bullish_fvg":
            deepest = lows[after].min() if third + 1 < n else top
            filled = (top - deepest) / gap
        else:
            deepest = highs[after].max() if third + 1 < n else bottom
            filled = (deepest - bottom) / gap
        filled = float(np.clip(filled, 0, 1))
        if filled >= MAX_FILLED:
            continue

        last = closes[-1]
        distance = 0.0 if bottom <= last <= top else min(abs(last - bottom), abs(last - top)) / last
        found.append({
            "kind": kind,
            "score": float(min(1.0, 0.7 + 0.3 * gap / size_atr)),
            "start": first,
            "window": n - first,
            "zone": (float(bottom), float(top)),
            "filled": filled,
            "distance": float(distance),
            "label": (
                f"{FVG_NAMES[kind]} {bottom:.6g}–{top:.6g}, "
                f"заполнен на {filled:.0%}, "
                + ("цена внутри зоны" if distance == 0 else f"до зоны {distance:.1%}")
            ),
        })
    return sorted(found, key=lambda f: f["start"], reverse=True)


def find_fvg(df, lookback=LOOKBACK):
    """Самый значимый свежий незакрытый FVG (или None).
    Значимость: крупнее относительно ATR и ближе к текущей цене."""
    gaps = find_all_fvg(df, lookback)
    if not gaps:
        return None
    return max(gaps, key=lambda f: f["score"] - f["distance"])
