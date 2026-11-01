"""Smart Money Concepts: детекторы по правилам на свечах.

FVG (Fair Value Gap, имбаланс) — «дыра» в цене из трёх свечей:
  бычий:    low третьей свечи выше high первой — между ними цена не торговалась;
  медвежий: high третьей свечи ниже low первой.
Цена часто возвращается «закрыть» такую дыру, поэтому незакрытые FVG
работают как уровни — и свежие, и старые.
"""

import numpy as np
import pandas as pd

ATR_PERIOD = 14
MIN_GAP_ATR = 0.5          # дыра меньше половины среднего хода свечи — шум
LOOKBACK = 30              # свежий FVG: образовался за последние 30 свечей
OLD_LOOKBACK = 150         # старый незакрытый FVG интересен, только если цена рядом
NEAR_PRICE = 0.01          # «рядом» = не дальше 1% от зоны
MAX_FILLED = 0.99          # заполнен на 100% — закрыт

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


def find_all_fvg(df, lookback=LOOKBACK, include_filled=False):
    """Все FVG за последние lookback свечей, от свежих к старым.

    По умолчанию — только незакрытые. include_filled=True — все, у закрытых
    в "filled_at" номер свечи, на которой дыру закрыли (для истории)."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = atr(df).to_numpy()
    n = len(df)
    last = closes[-1]
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

        # как глубоко цена залезала в дыру после образования, свеча за свечой
        if kind == "bullish_fvg":
            depth = (top - np.minimum.accumulate(lows[third + 1:])) / gap
        else:
            depth = (np.maximum.accumulate(highs[third + 1:]) - bottom) / gap
        depth = np.clip(depth, 0, 1)
        filled = float(depth[-1]) if len(depth) else 0.0
        closed = np.where(depth >= MAX_FILLED)[0]
        filled_at = int(third + 1 + closed[0]) if len(closed) else None
        if filled_at is not None and not include_filled:
            continue

        distance = 0.0 if bottom <= last <= top else min(abs(last - bottom), abs(last - top)) / last
        status = "закрыт" if filled_at is not None else f"заполнен на {filled:.0%}"
        where = "цена внутри зоны" if distance == 0 else f"до зоны {distance:.1%}"
        found.append({
            "kind": kind,
            "score": float(min(1.0, 0.7 + 0.3 * gap / size_atr)),
            "start": first,
            "end": third,
            "window": n - first,
            "zone": (float(bottom), float(top)),
            "filled": filled,
            "filled_at": filled_at,
            "distance": float(distance),
            "label": f"{FVG_NAMES[kind]} {bottom:.6g}–{top:.6g}, {status}, {where}",
        })
    return sorted(found, key=lambda f: f["start"], reverse=True)


def find_fvg(df):
    """Самый значимый незакрытый FVG для алерта (или None):
    свежий (последние 30 свечей) или старый (до 150 свечей), к которому подошла цена.
    Значимость: крупнее относительно ATR и ближе к текущей цене."""
    n = len(df)
    candidates = [
        f for f in find_all_fvg(df, OLD_LOOKBACK)
        if f["start"] >= n - LOOKBACK - 2 or f["distance"] <= NEAR_PRICE
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda f: f["score"] - f["distance"])
    if best["start"] < n - LOOKBACK - 2:
        best["label"] = "старый " + best["label"]
    return best
