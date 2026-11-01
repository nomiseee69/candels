"""Фигуры из линий тренда: треугольники, клинья, каналы.

На рынке: находим локальные максимумы и минимумы, проводим через них прямые
(np.polyfit) и смотрим, как прямые ведут себя: сходятся, параллельны, куда наклонены.
На рисунке пользователя: то же самое по двум нарисованным линиям — так бот
понимает, какую именно фигуру тот имел в виду.
"""

import numpy as np

PIVOT_WINDOW = 3            # экстремум = выше/ниже 3 свечей с каждой стороны
MIN_PIVOTS = 2              # минимум касаний каждой линии
MAX_PIVOTS = 4              # берём последние касания — старые уже не важны
FLAT_SLOPE = 0.15           # наклон за всё окно меньше 15% высоты фигуры — горизонталь
FLAT_ANGLE = 4              # на рисунке: наклон меньше 4° — горизонталь
CONVERGE_RATIO = 0.7        # к концу расстояние между линиями сжалось хотя бы на 30%
PARALLEL_RATIO = (0.8, 1.25)
MIN_SPAN = 0.4              # касания разнесены хотя бы на 40% окна
ERROR_WEIGHT = 4            # средняя невязка 5% высоты → оценка 0.8
BREAKOUT_TOLERANCE = 0.01   # цена может выйти за линию на 1% — ещё внутри фигуры
WINDOWS = (30, 50, 80, 120)

TRIANGLES = {"symmetric", "ascending", "descending"}
WEDGES = {"rising_wedge", "falling_wedge"}
CHANNELS = {"channel_up", "channel_down", "channel_flat"}
FAMILY_BY_PATTERN = {"Треугольник": TRIANGLES, "Клин": WEDGES, "Канал": CHANNELS}

KIND_NAMES = {
    "symmetric": "симметричный треугольник",
    "ascending": "восходящий треугольник",
    "descending": "нисходящий треугольник",
    "rising_wedge": "восходящий клин",
    "falling_wedge": "нисходящий клин",
    "channel_up": "восходящий канал",
    "channel_down": "нисходящий канал",
    "channel_flat": "боковой канал",
}


def _direction(value, flat):
    if abs(value) < flat:
        return "flat"
    return "up" if value > 0 else "down"


def classify(upper_dir, lower_dir, gap_start, gap_end):
    """Тип фигуры по направлениям линий и тому, как меняется расстояние между ними."""
    if gap_start <= 0:
        return None
    ratio = gap_end / gap_start
    if PARALLEL_RATIO[0] <= ratio <= PARALLEL_RATIO[1] and upper_dir == lower_dir:
        return f"channel_{upper_dir}"
    if ratio >= CONVERGE_RATIO:
        return None
    pair = (upper_dir, lower_dir)
    return {
        ("down", "up"): "symmetric",
        ("flat", "up"): "ascending",
        ("down", "flat"): "descending",
        ("up", "up"): "rising_wedge",
        ("down", "down"): "falling_wedge",
    }.get(pair)


# --- рисунок пользователя ---------------------------------------------------------------
def kind_from_drawing(lines):
    """Какую фигуру нарисовал пользователь двумя линиями (или None)."""
    if len(lines) != 2:
        return None
    a, b = lines
    left = max(a.x1, b.x1)
    right = min(a.x2, b.x2)
    if right <= left:
        return None
    # на картинке y растёт вниз: меньший y — выше
    top, bottom = (a, b) if a.y_at(left) < b.y_at(left) else (b, a)
    gap_start = bottom.y_at(left) - top.y_at(left)
    gap_end = bottom.y_at(right) - top.y_at(right)
    return classify(
        _direction(top.angle, FLAT_ANGLE),
        _direction(bottom.angle, FLAT_ANGLE),
        gap_start,
        gap_end,
    )


# --- рынок -------------------------------------------------------------------------------
def pivot_highs(values, window):
    """Локальные максимумы, устойчивые к «плоской вершине»: две соседние свечи
    с одинаковым high — частый случай. Точка не ниже соседей слева и строго выше
    соседей справа, поэтому из плоской вершины берётся её последняя свеча."""
    return [
        i for i in range(window, len(values) - window)
        if values[i] >= max(values[i - window:i]) and values[i] > max(values[i + 1:i + 1 + window])
    ]


def pivot_lows(values, window):
    return [
        i for i in range(window, len(values) - window)
        if values[i] <= min(values[i - window:i]) and values[i] < min(values[i + 1:i + 1 + window])
    ]


def _fit(xs, ys):
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = np.abs(np.asarray(ys) - (slope * np.asarray(xs) + intercept))
    return slope, intercept, residuals.mean()


def _analyze_window(highs, lows):
    """Ищет фигуру в одном окне свечей. Координаты — номера свечей внутри окна."""
    n = len(highs)
    maxima = pivot_highs(highs, PIVOT_WINDOW)[-MAX_PIVOTS:]
    minima = pivot_lows(lows, PIVOT_WINDOW)[-MAX_PIVOTS:]
    if len(maxima) < MIN_PIVOTS or len(minima) < MIN_PIVOTS:
        return None
    if (maxima[-1] - maxima[0]) < n * MIN_SPAN or (minima[-1] - minima[0]) < n * MIN_SPAN:
        return None

    up_slope, up_icpt, up_err = _fit(maxima, [highs[i] for i in maxima])
    lo_slope, lo_icpt, lo_err = _fit(minima, [lows[i] for i in minima])

    x0 = min(maxima[0], minima[0])
    x1 = n - 1
    height = (up_slope * x0 + up_icpt) - (lo_slope * x0 + lo_icpt)
    if height <= 0:
        return None
    gap_end = (up_slope * x1 + up_icpt) - (lo_slope * x1 + lo_icpt)

    span = x1 - x0
    kind = classify(
        _direction(up_slope * span / height, FLAT_SLOPE),
        _direction(lo_slope * span / height, FLAT_SLOPE),
        height,
        gap_end,
    )
    if kind is None:
        return None

    error = (up_err + lo_err) / 2 / height
    score = float(np.clip(1 - ERROR_WEIGHT * error, 0, 1))
    return {
        "kind": kind,
        "score": score,
        "upper": (x0, up_slope * x0 + up_icpt, x1, up_slope * x1 + up_icpt),
        "lower": (x0, lo_slope * x0 + lo_icpt, x1, lo_slope * x1 + lo_icpt),
    }


def _still_inside(found, last_close):
    upper_end = found["upper"][3]
    lower_end = found["lower"][3]
    return lower_end * (1 - BREAKOUT_TOLERANCE) <= last_close <= upper_end * (1 + BREAKOUT_TOLERANCE)


def find_line_pattern(df, family, preferred=None, windows=WINDOWS):
    """Лучшая фигура нужного семейства, которая формируется прямо сейчас (или None).

    family — набор типов (TRIANGLES, WEDGES, CHANNELS);
    preferred — тип с рисунка пользователя: совпадение повышает оценку.
    """
    highs_all = df["high"].tolist()
    lows_all = df["low"].tolist()
    last_close = float(df["close"].iloc[-1])
    best = None

    for w in windows:
        if len(df) < w:
            continue
        found = _analyze_window(highs_all[-w:], lows_all[-w:])
        if found is None or found["kind"] not in family:
            continue
        if not _still_inside(found, last_close):
            continue
        if preferred:
            found["score"] = min(1.0, found["score"] + 0.05) if found["kind"] == preferred else found["score"] - 0.1

        start = len(df) - w
        x0, u0, x1, u1 = found["upper"]
        _, l0, _, l1 = found["lower"]
        found.update({
            "start": start + x0,          # фигура начинается с первого касания линий
            "window": w - x0,
            "upper": (start + x0, u0, start + x1, u1),
            "lower": (start + x0, l0, start + x1, l1),
        })
        # при почти равной оценке лучше длинное окно — оно захватывает фигуру целиком
        if best is None or found["score"] + 0.0005 * found["window"] > best["score"] + 0.0005 * best["window"]:
            best = found
    return best
