"""Боковик (Range, консолидация) и профиль объёма внутри него.

Трейдер видит боковик как ПРЯМОУГОЛЬНИК: цена много раз отталкивается от верхней
и нижней границы и никуда не уходит. А вертикальный профиль объёма (Volume Profile)
показывает, на каких ЦЕНАХ прошло больше всего объёма: в честном боковике самая
толстая полка (POC — Point of Control) стоит ПОСЕРЕДИНЕ — покупатели и продавцы
в равновесии. Одной картинкой это объяснить трудно, поэтому бот ищет боковик по правилам:

  1) коробка: верх и низ окна свечей (без самых длинных теней — берём 95% и 5% цен);
  2) почти все закрытия (≥ 90%) внутри коробки;
  3) у верхней и у нижней границы — хотя бы по 2 касания (свинги в крайних 20% высоты);
  4) цена ходит туда-сюда: пересекла середину коробки хотя бы 3 раза,
     и за всё окно почти не сдвинулась (нет тренда);
  5) высота коробки — от 2 до 12 средних свечей (ATR): не шум и не размашистые качели;
  6) профиль объёма: объём каждой свечи раскладываем по ценам от её low до high
     (24 «полки»); POC должен быть в средней половине коробки.

Алерт: цена сейчас внутри боковика — или только что (за 3 свечи) закрылась за его границей
(пробой вверх / вниз).
"""

import numpy as np

from structure import atr
from trendlines import pivot_highs, pivot_lows

WINDOWS = (20, 30, 45, 60, 90, 120)
EDGE_QUANTILE = 0.05       # границы коробки без 5% самых длинных теней
INSIDE_SHARE = 0.9         # ≥ 90% закрытий внутри коробки
EDGE = 0.2                 # касание = свинг в верхних / нижних 20% высоты
MIN_TOUCHES = 2
MIN_SPAN = 0.4             # касания разнесены хотя бы на 40% окна
MIN_CROSSINGS = 3          # цена пересекла середину коробки хотя бы 3 раза
MAX_DRIFT = 0.35           # от начала к концу окна цена сдвинулась не больше 35% высоты
HEIGHT_ATR = (2.0, 12.0)   # высота коробки в средних свечах
BINS = 24                  # «полки» профиля объёма
POC_CENTER = (0.25, 0.75)  # POC — в средней половине коробки
VALUE_AREA = 0.7           # зона стоимости — 70% объёма вокруг POC
RECENT = 3                 # пробой — за последние 3 свечи
HISTORY_STEP = 5


def volume_profile(highs, lows, volumes, bottom, top, bins=BINS):
    """Объём по ценовым полкам: объём свечи делим поровну между полками от её low до high.
    Возвращает (объёмы полок, цены середин полок)."""
    edges = np.linspace(bottom, top, bins + 1)
    profile = np.zeros(bins)
    for high, low, volume in zip(highs, lows, volumes):
        low, high = max(low, bottom), min(high, top)
        if high < low:
            continue
        first = min(int(np.searchsorted(edges, low, side="right")) - 1, bins - 1)
        last = min(int(np.searchsorted(edges, high, side="right")) - 1, bins - 1)
        first, last = max(first, 0), max(last, 0)
        profile[first:last + 1] += volume / (last - first + 1)
    return profile, (edges[:-1] + edges[1:]) / 2


def value_area(profile, centers, share=VALUE_AREA):
    """Зона стоимости: от POC расширяемся к более «толстой» соседней полке, пока не наберём 70%."""
    poc = int(np.argmax(profile))
    low = high = poc
    total, need = profile[poc], share * profile.sum()
    while total < need and (low > 0 or high < len(profile) - 1):
        below = profile[low - 1] if low > 0 else -1
        above = profile[high + 1] if high < len(profile) - 1 else -1
        if above >= below:
            high += 1
            total += above
        else:
            low -= 1
            total += below
    return float(centers[low]), float(centers[high])


def analyze_range(highs, lows, closes, volumes, size):
    """Боковик в одном окне свечей (номера — внутри окна) или None. size — средняя свеча (ATR)."""
    n = len(closes)
    top = float(np.quantile(highs, 1 - EDGE_QUANTILE))
    bottom = float(np.quantile(lows, EDGE_QUANTILE))
    height = top - bottom
    if height <= 0 or size <= 0 or not HEIGHT_ATR[0] <= height / size <= HEIGHT_ATR[1]:
        return None
    if np.mean((closes >= bottom) & (closes <= top)) < INSIDE_SHARE:
        return None
    if abs(closes[-1] - closes[0]) > MAX_DRIFT * height:
        return None
    middle = (top + bottom) / 2
    side = np.sign(closes - middle)
    side = side[side != 0]
    if np.count_nonzero(np.diff(side)) < MIN_CROSSINGS:
        return None

    window = max(2, n // 20)
    tops = [i for i in pivot_highs(list(highs), window) if highs[i] >= top - EDGE * height]
    bottoms = [i for i in pivot_lows(list(lows), window) if lows[i] <= bottom + EDGE * height]
    if len(tops) < MIN_TOUCHES or len(bottoms) < MIN_TOUCHES:
        return None
    if min(tops[-1] - tops[0], bottoms[-1] - bottoms[0]) < MIN_SPAN * n:
        return None

    poc = va = None
    if volumes is not None:
        profile, centers = volume_profile(highs, lows, volumes, bottom, top)
        if profile.sum() > 0:
            poc = float(centers[int(np.argmax(profile))])
            if not POC_CENTER[0] <= (poc - bottom) / height <= POC_CENTER[1]:
                return None                                  # объём скопился у края — это не равновесие
            va = value_area(profile, centers)
    touches = len(tops) + len(bottoms)
    return {
        "kind": "range",
        "box": (bottom, top),
        "poc": poc,
        "value_area": va,
        "touches": touches,
        "touch_points": [(i, float(highs[i])) for i in tops] + [(i, float(lows[i])) for i in bottoms],
        # 4 касания → 0.85, 8+ → 1.0; при равенстве длинный боковик важнее
        "score": float(min(1.0, 0.85 + 0.0375 * (touches - 4))),
        "start": 0,
        "window": n,
    }


def _arrays(df):
    volumes = df["volume"].to_numpy(dtype=float) if "volume" in df else None
    return (df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float), volumes, atr(df).to_numpy())


def _at(df, arrays, start, end):
    """Боковик на свечах start..end графика (координаты — номера свечей графика)."""
    highs, lows, closes, volumes, sizes = arrays
    found = analyze_range(highs[start:end + 1], lows[start:end + 1], closes[start:end + 1],
                          None if volumes is None else volumes[start:end + 1],
                          float(np.mean(sizes[start:end + 1])))
    if found is None:
        return None
    found["start"] = start
    found["end"] = end
    found["touch_points"] = [(x + start, y) for x, y in found["touch_points"]]
    return found


def _after(df, found, horizon):
    """Что было после боковика: первая свеча, закрывшаяся за коробкой, и куда."""
    bottom, top = found["box"]
    closes = df["close"].to_numpy(dtype=float)
    for i in range(found["end"] + 1, min(found["end"] + 1 + horizon, len(closes))):
        if closes[i] > top:
            return "up", i
        if closes[i] < bottom:
            return "down", i
    return None, None


def _finish(df, found):
    bottom, top = found["box"]
    last = float(df["close"].iloc[-1])
    found["distance"] = 0.0 if bottom <= last <= top else min(abs(last - bottom), abs(last - top)) / last
    length = found["end"] - found["start"] + 1
    text = f"боковик {bottom:.6g}–{top:.6g}, {length} свечей"
    if found.get("poc") is not None:
        text += f", POC {found['poc']:.6g} (середина объёма)"
    if found.get("breakout"):
        text += ", пробой " + ("вверх" if found["breakout"] == "up" else "вниз")
    elif found["distance"] == 0:
        text += ", цена внутри"
    found["label"] = text
    return found


def find_range(df, windows=WINDOWS):
    """Боковик «сейчас»: цена внутри него или только что пробила границу. Самый длинный."""
    n = len(df)
    arrays = _arrays(df)
    best = None
    for w in windows:
        for back in range(RECENT + 1):                       # окно кончается на последней свече или чуть раньше
            end = n - 1 - back
            start = end - w + 1
            if start < 0:
                continue
            found = _at(df, arrays, start, end)
            if found is None:
                continue
            breakout, at = _after(df, found, n)
            if back > 0 and breakout is None:
                continue                                      # всё ещё внутри — это окно с back = 0
            found["breakout"] = breakout
            found["window"] = n - start                        # до последней свечи — для разметки
            rank = (found["score"], w, -back)
            if best is None or rank > best[0]:
                best = (rank, found)
            break
    return _finish(df, best[1]) if best else None


def history_ranges(df, windows=WINDOWS, horizon=10):
    """Все боковики за период (без пересечений) и куда цена вышла из них."""
    from history import non_overlapping, outcome
    n = len(df)
    arrays = _arrays(df)
    found = []
    for w in windows:
        for end in range(w - 1, n, max(HISTORY_STEP, w // 6)):
            f = _at(df, arrays, end - w + 1, end)
            if f is not None:
                f["score"] += 0.0005 * w                     # из пересекающихся — более длинный
                found.append(f)
    found = non_overlapping(found)
    for f in found:
        f["score"] = min(1.0, f["score"])
        f["breakout"], _ = _after(df, f, horizon)
        f["outcome"] = outcome(df, f["end"], horizon)
        _finish(df, f)
    return found
