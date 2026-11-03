"""Smart Money Concepts: детекторы по правилам на свечах.

FVG (Fair Value Gap, имбаланс) — «дыра» в цене из трёх свечей:
  бычий:    low третьей свечи выше high первой — между ними цена не торговалась;
  медвежий: high третьей свечи ниже low первой.
Цена часто возвращается «закрыть» такую дыру, поэтому незакрытые FVG
работают как уровни — и свежие, и старые.

Значимый FVG — не любая дыра, а след крупного игрока. Нужны ВСЕ условия:
  1) средняя свеча — импульс: тело ≥ 1 ATR и занимает ≥ 60% свечи (displacement);
  2) импульс сломал структуру (BOS/CHoCH) — сразу или в 3 свечах после;
  3) дыра не мелкая: ≥ 0.75 ATR;
  4) на импульсе был объём: ≥ 1.2× среднего за 20 свечей (неликвидные часы отсеиваются).
Отработка значимого FVG — свеча ЗАКРЫЛАСЬ за его серединой (50%, consequent encroachment):
после этого зона считается использованной, даже если до конца её не закрыли.
"""

import numpy as np

from structure import atr, structure_events

ATR_PERIOD = 14
MIN_GAP_ATR = 0.5          # дыра меньше половины среднего хода свечи — шум
LOOKBACK = 30              # свежий FVG: образовался за последние 30 свечей
OLD_LOOKBACK = 1000        # старый незакрытый FVG интересен, только если цена рядом (любой давности)
NEAR_PRICE = 0.01          # «рядом» = не дальше 1% от зоны
MAX_FILLED = 0.99          # заполнен на 100% — закрыт
IMPULSE_BODY_ATR = 1.0     # тело импульсной свечи ≥ 1 ATR
IMPULSE_BODY_SHARE = 0.6   # и тело ≥ 60% всей свечи (маленькие тени)
SIGNIFICANT_GAP_ATR = 0.75 # значимая дыра ≥ 0.75 ATR
VOLUME_MIN = 1.2           # объём импульса ≥ 1.2× среднего за 20 свечей
BREAK_REACH = 3            # слом структуры засчитываем, если он на импульсе или в 3 свечах после

FVG_NAMES = {"bullish_fvg": "бычий FVG", "bearish_fvg": "медвежий FVG"}


def _breaks_by_candle(df):
    """Номер свечи → направление слома структуры на ней ("up" / "down")."""
    return {e["end"]: e["direction"] for e in structure_events(df)}


def _volume_ok(df):
    """Был ли на свече объём выше обычного. Нет столбца объёма (тесты) — не проверяем."""
    if "volume" not in df:
        return None
    volume = df["volume"]
    average = volume.rolling(20, min_periods=5).mean().shift(1)    # средний объём ДО свечи
    return (volume >= VOLUME_MIN * average).fillna(True).to_numpy()


def find_all_fvg(df, lookback=LOOKBACK, include_filled=False, significant_only=False):
    """Все FVG за последние lookback свечей, от свежих к старым.

    По умолчанию — только незакрытые. include_filled=True — все, у закрытых
    в "filled_at" номер свечи, на которой дыру закрыли (для истории).
    significant_only=True — только значимые (см. описание модуля)."""
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = atr(df).to_numpy()
    breaks = _breaks_by_candle(df)
    volume_ok = _volume_ok(df)
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

        # значимость: импульс + слом структуры + не мелкая дыра + объём
        middle = i - 1
        body = abs(closes[middle] - opens[middle])
        candle = highs[middle] - lows[middle]
        base_atr = atrs[first]                              # ход ДО импульса
        impulse = body >= IMPULSE_BODY_ATR * base_atr and candle > 0 and body / candle >= IMPULSE_BODY_SHARE
        direction = "up" if kind == "bullish_fvg" else "down"
        broke = any(breaks.get(c) == direction for c in range(middle, min(third + BREAK_REACH, n - 1) + 1))
        wide = gap >= SIGNIFICANT_GAP_ATR * size_atr
        volume = bool(volume_ok[middle]) if volume_ok is not None else True
        significant = bool(impulse and broke and wide and volume)
        if significant_only and not significant:
            continue
        reasons = ["импульс, слом структуры"] + (["объём"] if volume_ok is not None else [])

        # как глубоко цена залезала в дыру после образования, свеча за свечой
        if kind == "bullish_fvg":
            depth = (top - np.minimum.accumulate(lows[third + 1:])) / gap
            beyond_middle = closes[third + 1:] < (bottom + top) / 2
        else:
            depth = (np.maximum.accumulate(highs[third + 1:]) - bottom) / gap
            beyond_middle = closes[third + 1:] > (bottom + top) / 2
        depth = np.clip(depth, 0, 1)
        filled = float(depth[-1]) if len(depth) else 0.0
        closed = np.where(depth >= MAX_FILLED)[0]
        filled_at = int(third + 1 + closed[0]) if len(closed) else None
        worked_out = False
        if significant_only:                                # отработан: закрытие за серединой
            ce = np.flatnonzero(beyond_middle)
            if len(ce) and (filled_at is None or third + 1 + ce[0] < filled_at):
                filled_at, worked_out = int(third + 1 + ce[0]), True
        if filled_at is not None and not include_filled:
            continue

        distance = 0.0 if bottom <= last <= top else min(abs(last - bottom), abs(last - top)) / last
        if filled_at is None:
            status = f"заполнен на {filled:.0%}"
        else:
            status = "отработан (закрытие за серединой)" if worked_out else "закрыт"
        where = "цена внутри зоны" if distance == 0 else f"до зоны {distance:.1%}"
        tags = f" ({', '.join(reasons)})" if significant else ""
        score = 0.7 + 0.3 * min(1.0, gap / size_atr)
        if significant:
            score = max(score, 0.9)
        found.append({
            "kind": kind,
            "score": float(score),
            "start": first,
            "end": third,
            "window": n - first,
            "zone": (float(bottom), float(top)),
            "filled": filled,
            "filled_at": filled_at,
            "distance": float(distance),
            "significant": significant,
            "broke_structure": bool(broke and significant),
            "label": f"{FVG_NAMES[kind]}{tags} {bottom:.6g}–{top:.6g}, {status}, {where}",
        })
    return sorted(found, key=lambda f: f["start"], reverse=True)


def find_fvg(df, significant_only=False, choose=None):
    """Лучший незакрытый FVG для алерта (или None):
    свежий (последние 30 свечей) или старый (до 150 свечей), к которому подошла цена.
    Выше оценка — крупнее относительно ATR, значимый и ближе к текущей цене."""
    n = len(df)
    candidates = [
        f for f in find_all_fvg(df, OLD_LOOKBACK, significant_only=significant_only)
        if f["start"] >= n - LOOKBACK - 2 or f["distance"] <= NEAR_PRICE
    ]
    if not candidates:
        return None
    # choose — оценка «похоже на твой пример» (fingerprint): из подходящих берём самый похожий
    best = max(candidates, key=choose or (lambda f: f["score"] - f["distance"]))
    if best["start"] < n - LOOKBACK - 2:
        best["label"] = "старый " + best["label"]
    return best
