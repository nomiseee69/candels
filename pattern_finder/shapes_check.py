"""Проверка формы по правилам трейдера: двойное дно / вершина, голова и плечи.

Похожесть ломаной на эталон «W» — только первый фильтр: так «двойным дном» становится
любая пила. Трейдер смотрит на конкретные точки, поэтому бот после сравнения формы
находит их на свечах и проверяет:

ДВОЙНОЕ ДНО
  • два минимума (дно 1 в первой половине, дно 2 во второй), между ними ≥ 30% окна;
  • дно примерно на одном уровне: разница ≤ 0.6 средней свечи (ATR) или ≤ 15% высоты фигуры;
  • «шея» — самый высокий хай между днами; высота фигуры (шея − дно) ≥ 2.5 ATR;
  • перед первым дном цена пришла СВЕРХУ (падение ≥ 70% высоты фигуры);
  • после второго дна цена отскочила хотя бы на половину высоты;
  • если цена закрылась выше шеи — фигура подтверждена («шея пробита»).
ДВОЙНАЯ ВЕРШИНА — то же в зеркале.

ГОЛОВА И ПЛЕЧИ
  • три вершины: голова в середине окна выше обоих плеч (заметно — ≥ 0.5 ATR и ≥ 10% высоты);
  • плечи примерно на одном уровне (разница ≤ 35% высоты), оба выше линии шеи на ≥ 1 ATR;
  • шея — линия через два минимума между плечами и головой; высота (голова − шея) ≥ 3 ATR;
  • подтверждение — закрытие ниже шеи.
ПЕРЕВЁРНУТАЯ ГИП — то же в зеркале.

Результат — точки фигуры (для рисования ломаной на графике) и уровень шеи.
"""

import numpy as np

from structure import atr
from trendlines import pivot_highs

MIN_GAP = 0.3            # дно 2 позже дна 1 хотя бы на 30% окна
SAME_LEVEL_ATR = 0.6
SAME_LEVEL_SHARE = 0.15
MIN_HEIGHT_ATR = 2.5
ARRIVAL = 0.7            # пришли к первому дну сверху: падение ≥ 70% высоты
REBOUND = 0.5            # после второго дна отскок ≥ 50% высоты
HEAD_ATR = 0.5
HEAD_SHARE = 0.1
SHOULDERS_SHARE = 0.35
HS_HEIGHT_ATR = 3.0

MIRRORED = {"Двойная вершина": "Двойное дно", "Перевёрнутая ГиП": "Голова и плечи"}


def _double_bottom(highs, lows, closes, size, before=None):
    n = len(lows)
    half = n // 2
    i1 = int(np.argmin(lows[:half]))
    i2 = half + int(np.argmin(lows[half:]))
    if i2 - i1 < max(5, MIN_GAP * n) or i1 < 0.1 * n:
        return None
    k = i1 + int(np.argmax(highs[i1:i2 + 1]))
    neck, bottom = highs[k], min(lows[i1], lows[i2])
    height = neck - bottom
    if height < MIN_HEIGHT_ATR * size:
        return None
    if abs(lows[i1] - lows[i2]) > max(SAME_LEVEL_ATR * size, SAME_LEVEL_SHARE * height):
        return None
    came_from = max(highs[:i1 + 1].max(), before.max() if before is not None and len(before) else -np.inf)
    if came_from - lows[i1] < ARRIVAL * height:          # к первому дну пришли сверху (можно и до окна)
        return None
    if i2 + 1 >= n or highs[i2 + 1:].max() - lows[i2] < REBOUND * height:
        return None
    start_y = float(highs[:i1 + 1].max())
    points = [(int(np.argmax(highs[:i1 + 1])), start_y), (i1, float(lows[i1])), (k, float(neck)),
              (i2, float(lows[i2])), (n - 1, float(closes[-1]))]
    confirmed = bool((closes[i2 + 1:] > neck).any())
    return {"points": points, "neckline": (i1, float(neck), n - 1, float(neck)), "confirmed": confirmed,
            "height": float(height)}


def _head_and_shoulders(highs, lows, closes, size, before=None):
    n = len(highs)
    window = max(2, n // 20)
    peaks = [p for p in pivot_highs(list(highs), window)]
    middle = [p for p in peaks if 0.2 * n <= p <= 0.8 * n]
    if not middle:
        return None
    head = max(middle, key=lambda p: highs[p])
    # плечо — вершина, от которой до головы цена заметно опускалась (иначе это просто склон головы)
    lefts = [p for p in peaks if p < head - 2 and lows[p:head].min() < highs[p] - size]
    rights = [p for p in peaks if p > head + 2 and lows[head:p].min() < highs[p] - size]
    if not lefts or not rights:
        return None
    left = max(lefts, key=lambda p: highs[p])
    right = max(rights, key=lambda p: highs[p])
    t1 = left + int(np.argmin(lows[left:head + 1]))
    t2 = head + int(np.argmin(lows[head:right + 1]))
    if min(head - left, right - head, t1 - left, head - t1, t2 - head, right - t2) < 2:
        return None
    neck = (lows[t1] + lows[t2]) / 2
    height = highs[head] - neck
    shoulders = max(highs[left], highs[right])
    if height < HS_HEIGHT_ATR * size:
        return None
    if highs[head] - shoulders < max(HEAD_ATR * size, HEAD_SHARE * height):
        return None
    if abs(highs[left] - highs[right]) > SHOULDERS_SHARE * height:
        return None
    if min(highs[left], highs[right]) - max(lows[t1], lows[t2]) < size:
        return None
    slope = (lows[t2] - lows[t1]) / (t2 - t1)
    neck_end = lows[t1] + slope * (n - 1 - t1)
    points = [(left, float(highs[left])), (t1, float(lows[t1])), (head, float(highs[head])),
              (t2, float(lows[t2])), (right, float(highs[right])), (n - 1, float(closes[-1]))]
    confirmed = bool(any(closes[x] < lows[t1] + slope * (x - t1) for x in range(right + 1, n)))
    return {"points": points, "neckline": (t1, float(lows[t1]), n - 1, float(neck_end)),
            "confirmed": confirmed, "height": float(height)}


CHECKS = {"Двойное дно": _double_bottom, "Голова и плечи": _head_and_shoulders}


def validate(name, df, start, window, sizes=None):
    """Проверка фигуры name на свечах start..start+window. Возвращает точки (номера свечей
    графика), шею и подтверждение — или None, если по правилам это не та фигура.
    Для паттернов без правил (нет в CHECKS) — {} (проверять нечего)."""
    base = MIRRORED.get(name, name)
    check = CHECKS.get(base)
    if check is None:
        return {}
    end = min(start + window, len(df))
    if end - start < 10:
        return None
    part = df.iloc[start:end]
    highs, lows, closes = (part[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    sizes = atr(df).to_numpy() if sizes is None else sizes          # ATR можно посчитать заранее
    size = float(np.mean(sizes[start:end])) or 1e-9
    mirror = name in MIRRORED
    lead = df.iloc[max(0, start - window // 2):start]   # что было ДО окна (откуда пришла цена)
    before_highs, before_lows = lead["high"].to_numpy(dtype=float), lead["low"].to_numpy(dtype=float)
    if mirror:                                           # вершина = дно вверх ногами
        highs, lows, closes = -lows, -highs, -closes
        before_highs = -before_lows
    found = check(highs, lows, closes, size, before_highs)
    if found is None:
        return None
    sign = -1 if mirror else 1
    found["points"] = [(x + start, sign * y) for x, y in found["points"]]
    x0, y0, x1, y1 = found["neckline"]
    found["neckline"] = (x0 + start, sign * y0, x1 + start, sign * y1)
    return found


def shape_label(name, found):
    neck = found["neckline"][1]
    text = f"{name.lower()}: шея {neck:.6g}"
    return text + (", шея пробита — фигура подтверждена" if found.get("confirmed") else ", ждём пробоя шеи")


def read_shape(name, path):
    """Двойное дно / вершина, ГиП на СКРИНШОТЕ: читаем свечи и ищем фигуру по точкам на
    любом участке скрина (фигура может занимать не весь скрин). (свечи, фигура) или None."""
    from structure import atr
    from vision import extract_candles
    df = extract_candles(path)
    if df is None or len(df) < 20:
        return None
    n = len(df)
    sizes = atr(df).to_numpy()
    best = None
    for start in range(0, max(1, n - 20), 2):
        for end in range(n, start + 19, -3):
            found = validate(name, df, start, end - start, sizes)
            if found and (best is None or found["height"] > best[1]["height"]):
                best = (start, found)
    return (df, best[1]) if best else None
