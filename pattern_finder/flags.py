"""Флаг и вымпел — фигуры продолжения тренда.

Обе начинаются с ФЛАГШТОКА — резкого движения на несколько свечей (не меньше 5 средних
свечей ATR). Потом цена «отдыхает»:
  • ФЛАГ — небольшой канал против движения (или горизонтальный): линии почти параллельны;
  • ВЫМПЕЛ — маленький сходящийся треугольник.
Отдых короткий (не длиннее 3 флагштоков) и неглубокий: цена откатывает не больше
половины флагштока. Ждём выхода по направлению флагштока.

Линии флага короткие (8–30 свечей), поэтому строим их не по свингам, как у больших
треугольников, а прямой через все хаи (верх) и все лои (низ) — методом наименьших
квадратов — и сдвигаем так, чтобы свечи были внутри. Касание — свеча у линии
(ближе 25% средней свечи).
"""

import numpy as np

from structure import atr

POLES = (4, 6, 8, 12, 16)          # длина флагштока, свечей
BODIES = (6, 8, 12, 16, 22, 30)    # длина флага / вымпела, свечей
POLE_ATR = 5.0                     # флагшток — движение ≥ 5 ATR
MAX_RETRACE = 0.5                  # откат не больше 50% флагштока
MAX_BODY_HEIGHT = 0.5              # ширина флага не больше половины флагштока
MAX_BODY_POLES = 3                 # флаг не длиннее 3 флагштоков
TOUCH = 0.25                       # касание — свеча ближе 25% ATR к линии
MIN_TOUCHES = 2
PARALLEL = (0.6, 1.4)              # флаг: ширина в конце / в начале
CONVERGE = 0.6                     # вымпел: к концу сузился хотя бы на 40%
AGAINST = 0.25                     # флаг не должен идти по флагштоку дальше 25% своей ширины
INSIDE = 0.02                      # цена может выйти за линию на 2% ширины флагштока
RECENT = 3
HISTORY_STEP = 3

NAMES = {
    "bullish_flag": "бычий флаг", "bearish_flag": "медвежий флаг",
    "bullish_pennant": "бычий вымпел", "bearish_pennant": "медвежий вымпел",
}


def _edge(values, upper):
    """Граница флага: прямая через верхние точки (upper) или нижние, сдвинутая так,
    чтобы все свечи были внутри. Сначала прямая через все точки, потом — только через те,
    что выше (ниже) неё: так линия идёт по вершинам, а не по середине «пилы»."""
    x = np.arange(len(values))
    keep = np.ones(len(values), dtype=bool)
    for _ in range(2):
        slope, icpt = np.polyfit(x[keep], values[keep], 1)
        residual = values - (slope * x + icpt)
        side = residual >= 0 if upper else residual <= 0
        if side.sum() < 2:
            break
        keep = side
    slope, icpt = np.polyfit(x[keep], values[keep], 1)
    residual = values - (slope * x + icpt)
    icpt += residual.max() if upper else residual.min()
    return float(slope), float(icpt)


def analyze_body(highs, lows, size):
    """Линии флага/вымпела на коротком отрезке: (тип "flag"/"pennant", верх, низ, касания, точки) или None."""
    n = len(highs)
    x = np.arange(n)
    su, iu = _edge(highs, True)
    sl, il = _edge(lows, False)
    upper, lower = su * x + iu, sl * x + il
    width0, width1 = upper[0] - lower[0], upper[-1] - lower[-1]
    if width0 <= 0 or width1 <= 0:
        return None
    up_touch = np.flatnonzero(upper - highs <= TOUCH * size)
    low_touch = np.flatnonzero(lows - lower <= TOUCH * size)
    if len(up_touch) < MIN_TOUCHES or len(low_touch) < MIN_TOUCHES:
        return None
    if min(up_touch[-1] - up_touch[0], low_touch[-1] - low_touch[0]) < 0.4 * n:
        return None                                   # касания разнесены по флагу, а не в одном углу
    ratio = width1 / width0
    if PARALLEL[0] <= ratio <= PARALLEL[1]:
        shape = "flag"
    elif ratio <= CONVERGE and su <= 0 <= sl:           # верх вниз, низ вверх — сходится
        shape = "pennant"
    else:
        return None
    points = [(int(i), float(highs[i])) for i in up_touch] + [(int(i), float(lows[i])) for i in low_touch]
    return shape, (0, float(upper[0]), n - 1, float(upper[-1])), (0, float(lower[0]), n - 1, float(lower[-1])), \
        len(up_touch) + len(low_touch), points


def _at(arrays, shape_wanted, body_start, body_end):
    """Флаг/вымпел с телом body_start..body_end (номера свечей графика) или None."""
    highs, lows, closes, sizes = arrays
    size = float(np.mean(sizes[body_start:body_end + 1]))
    # быстрая проверка до подбора линий: был ли вообще резкий ход перед телом
    if not any(body_start - pole >= 0 and abs(closes[body_start] - closes[body_start - pole]) >= POLE_ATR * size
               for pole in POLES):
        return None
    body = analyze_body(highs[body_start:body_end + 1], lows[body_start:body_end + 1], size)
    if body is None or body[0] != shape_wanted:
        return None
    shape, upper, lower, touches, points = body
    body_len = body_end - body_start + 1
    candidates = []
    for pole in POLES:
        p0 = body_start - pole
        if p0 < 0 or body_len > MAX_BODY_POLES * pole:
            continue
        start_price, end_price = closes[p0], closes[body_start]
        move = end_price - start_price
        if abs(move) < POLE_ATR * size:
            continue
        up = move > 0
        pole_low = float(lows[p0:body_start + 1].min())
        pole_high = float(highs[p0:body_start + 1].max())
        height = pole_high - pole_low
        if upper[1] - lower[1] > MAX_BODY_HEIGHT * height:
            continue
        body_lows, body_highs = lows[body_start:body_end + 1], highs[body_start:body_end + 1]
        if up and body_lows.min() < pole_high - MAX_RETRACE * height:
            continue
        if not up and body_highs.max() > pole_low + MAX_RETRACE * height:
            continue
        middle_slope = ((upper[3] + lower[3]) - (upper[1] + lower[1])) / 2
        if shape == "flag" and (middle_slope if up else -middle_slope) > AGAINST * (upper[1] - lower[1]):
            continue                                      # «флаг» продолжает флагшток — это не отдых
        candidates.append((move, p0, pole_low, pole_high, up))
    if not candidates:
        return None
    # флагшток — самый большой ход среди «крутых» (≥ 70% самой большой скорости за свечу):
    # длинное окно с медленным подходом к флагштоку не считается его частью
    steep = max(abs(c[0]) / (body_start - c[1]) for c in candidates)
    move, p0, pole_low, pole_high, up = max(
        (c for c in candidates if abs(c[0]) / (body_start - c[1]) >= 0.7 * steep), key=lambda c: abs(c[0]))
    kind = ("bullish_" if up else "bearish_") + shape
    retrace = ((pole_high - lows[body_start:body_end + 1].min()) if up
               else (highs[body_start:body_end + 1].max() - pole_low)) / (pole_high - pole_low)
    shift = lambda line: (line[0] + body_start, line[1], line[2] + body_start, line[3])
    return {
        "kind": kind,
        "direction": "up" if up else "down",
        "upper": shift(upper),
        "lower": shift(lower),
        "pole": (p0, float(pole_low if up else pole_high), body_start, float(pole_high if up else pole_low)),
        "touches": touches,
        "touch_points": [(x + body_start, y) for x, y in points],
        "violations": 0,
        "retrace": float(retrace),
        "pole_move": float(move / closes[p0]),
        # 4 касания → 0.85, 8+ → 1.0
        "score": float(min(1.0, 0.85 + 0.0375 * (touches - 4))),
        "start": p0,
        "end": body_end,
        "window": body_end - p0 + 1,
    }


def _arrays(df):
    return (df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float), atr(df).to_numpy())


def _label(found, breakout=None):
    retrace = found["retrace"]
    depth = "неглубокий" if retrace < 0.38 else "средний" if retrace < 0.62 else "глубокий"
    text = (f"{NAMES[found['kind']]}: флагшток {'вверх' if found['pole_move'] > 0 else 'вниз'} за "
            f"{found['pole'][2] - found['pole'][0]} свечей, откат {depth}")
    if breakout:
        text += ", выход " + ("вверх" if breakout == "up" else "вниз")
    found["label"] = text
    return found


def find_flag(df, shape, extra=None):
    """Флаг (shape="flag") или вымпел ("pennant"), который формируется прямо сейчас:
    тело кончается на последних свечах, цена ещё внутри (или только что вышла по флагштоку)."""
    n = len(df)
    arrays = _arrays(df)
    closes = arrays[2]
    best = None
    for body in BODIES:
        for back in range(RECENT + 1):
            end = n - 1 - back
            found = _at(arrays, shape, end - body + 1, end) if end - body + 1 > 0 else None
            if found is None:
                continue
            breakout = None
            for i in range(end + 1, n):                       # что было после тела
                over = closes[i] > _line(found["upper"], i) + INSIDE * abs(found["pole"][3] - found["pole"][1])
                under = closes[i] < _line(found["lower"], i) - INSIDE * abs(found["pole"][3] - found["pole"][1])
                if over or under:
                    breakout = "up" if over else "down"
                    break
            if back > 0 and breakout != found["direction"]:
                continue                                      # вышли против флагштока — фигура сломана
            found["window"] = n - found["start"]
            bonus = extra(df, {**found, "breakout": breakout}) if extra else 0     # описание словами
            rank = (bonus, found["score"], body, -back)
            if best is None or rank > best[0]:
                best = (rank, _label(found, breakout))
            break
    return best[1] if best else None


def _line(line, x):
    x0, y0, x1, y1 = line
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0


def history_flags(df, shape, horizon=10):
    """Все флаги/вымпелы за период и куда цена вышла после них."""
    from history import non_overlapping, outcome
    n = len(df)
    arrays = _arrays(df)
    closes = arrays[2]
    found = []
    for body in BODIES:
        for end in range(body, n, HISTORY_STEP):
            f = _at(arrays, shape, end - body + 1, end)
            if f is not None:
                found.append(f)
    found = non_overlapping(found)
    for f in found:
        f["breakout"] = None
        for i in range(f["end"] + 1, min(f["end"] + 1 + horizon, n)):
            if closes[i] > _line(f["upper"], i):
                f["breakout"] = "up"
                break
            if closes[i] < _line(f["lower"], i):
                f["breakout"] = "down"
                break
        f["outcome"] = outcome(df, f["end"], horizon)
        _label(f, f["breakout"])
    return found
