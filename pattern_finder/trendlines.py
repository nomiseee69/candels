"""Фигуры из линий тренда: треугольники, клинья, каналы.

На рынке: находим локальные максимумы и минимумы, проводим через них прямые
(np.polyfit) и смотрим, как прямые ведут себя: сходятся, параллельны, куда наклонены.
На рисунке пользователя: то же самое по двум нарисованным линиям — так бот
понимает, какую именно фигуру тот имел в виду.
"""

import numpy as np

PIVOT_WINDOW = 3            # экстремум = выше/ниже 3 свечей с каждой стороны (в длинных окнах — больше)
MIN_PIVOTS = 2              # минимум касаний каждой линии
TOUCH_TOL = 0.5             # касание = свинг не дальше половины средней свечи от линии
MIN_GAP_END = 0.1           # к концу окна линии ещё не сошлись (ширина ≥ 10% начальной)
FLAT_SLOPE = 0.15           # наклон за всё окно меньше 15% высоты фигуры — горизонталь
FLAT_ANGLE = 4              # на рисунке: наклон меньше 4° — горизонталь
CONVERGE_RATIO = 0.7        # к концу расстояние между линиями сжалось хотя бы на 30%
PARALLEL_RATIO = (0.8, 1.25)
CHANNEL_TOUCHES = 5         # канал: всего не меньше 5 касаний (2 + 3), по 2 на линию — MIN_PIVOTS
CHANNEL_VIOLATIONS = 2      # и не больше 2 закрытий / проколов за линиями
MIN_SPAN = 0.4              # касания разнесены хотя бы на 40% окна
BREAKOUT_TOLERANCE = 0.01   # цена может выйти за линию на 1% — ещё внутри фигуры
WINDOWS = (30, 50, 80, 120, 180, 250, 350, 500, 700)

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
    # флаги и вымпелы (flags.py) — тоже фигуры из двух линий
    "bullish_flag": "бычий флаг", "bearish_flag": "медвежий флаг",
    "bullish_pennant": "бычий вымпел", "bearish_pennant": "медвежий вымпел",
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
    # наклон меряем по ВСЕЙ фигуре — от начала более длинной линии: короткая вторая линия
    # (её часто рисуют позже) иначе делает наклонную сторону «горизонтальной»
    common = left                                  # где видны обе линии — там понятно, какая сверху
    left = min(a.x1, b.x1)
    # на картинке y растёт вниз: меньший y — выше
    top, bottom = (a, b) if a.y_at(common) < b.y_at(common) else (b, a)
    gap_start = bottom.y_at(left) - top.y_at(left)
    gap_end = bottom.y_at(right) - top.y_at(right)
    if gap_start <= 0:
        return None
    # наклон — как на рынке: насколько линия поднялась за фигуру относительно её высоты
    # (а не в градусах: градусы зависят от того, как сжат скрин по ширине)
    rise = lambda ln: (ln.y_at(left) - ln.y_at(right)) / gap_start      # y вниз — переворачиваем
    return classify(
        _direction(rise(top), FLAT_SLOPE),
        _direction(rise(bottom), FLAT_SLOPE),
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


def drawn_span_has_swings(df, lines):
    """A two-sided formation needs an observed swing high and low under the drawing."""
    first = max(0, int(np.floor(min(line["x0"] for line in lines))))
    last = min(len(df) - 1, int(np.ceil(max(line["x1"] for line in lines))))
    observed = df.iloc[first:last + 1]
    edges = [line[key] for line in lines for key in ("y0", "y1")]
    if observed.empty or observed.high.max() < min(edges) or observed.low.min() > max(edges):
        return False
    highs = pivot_highs(df.high.tolist(), PIVOT_WINDOW)
    lows = pivot_lows(df.low.tolist(), PIVOT_WINDOW)
    return any(first <= i <= last for i in highs) and any(first <= i <= last for i in lows)


def _pivot_window(n):
    """Чем длиннее окно, тем крупнее свинги: в окне из 250 свечей нужны экстремумы
    среди 10 соседей, а не 3, — иначе каждый мелкий зубец станет «касанием»."""
    return max(PIVOT_WINDOW, n // 25)


def _best_boundary(pivots, values, closes, upper, tol):
    """Лучшая граница фигуры: прямая через два свинга, которую цена уважает.

    Перебираем все пары свингов (через две точки прямая проходит всегда — поэтому
    одной «точности прохождения» мало) и для каждой прямой считаем:
      касания  — сколько свингов лежат на ней (не дальше tol);
      нарушения — сколько свечей ЗАКРЫЛИСЬ за ней и сколько свингов проткнули её тенью.
    Хорошая граница — много касаний и почти ни одного нарушения."""
    piv = np.asarray(pivots)
    if len(piv) < 2:
        return None
    a, b = np.triu_indices(len(piv), 1)
    n = len(values)
    keep = piv[b] - piv[a] >= MIN_SPAN * (n - piv[a])
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return None
    values = np.asarray(values, dtype=float)
    closes = np.asarray(closes, dtype=float)
    xa, xb = piv[a], piv[b]
    slope = (values[xb] - values[xa]) / (xb - xa)
    icpt = values[xa] - slope * xa
    sign = 1.0 if upper else -1.0                           # «за линией» = выше верхней / ниже нижней

    at_pivots = slope[:, None] * piv[None, :] + icpt[:, None]
    outside = sign * (values[piv][None, :] - at_pivots)     # >0 — свинг за линией
    touch = np.abs(outside) <= tol
    first = np.where(touch, piv[None, :], n).min(axis=1)    # фигура начинается с первого касания
    last = np.where(touch, piv[None, :], -1).max(axis=1)
    after_first = piv[None, :] >= first[:, None]
    touches = (touch & after_first).sum(axis=1)
    pierced = ((outside > 2 * tol) & after_first).sum(axis=1)

    x = np.arange(n)
    line = slope[:, None] * x[None, :] + icpt[:, None]
    closed_out = (sign * (closes[None, :] - line) > tol) & (x[None, :] >= first[:, None])
    violations = closed_out.sum(axis=1) + pierced

    quality = 2 * touches - 3 * violations + (last - first) / n
    i = int(np.argmax(quality))
    touched = piv[touch[i] & after_first[i]]
    return {
        "slope": float(slope[i]), "icpt": float(icpt[i]),
        "touches": int(touches[i]), "violations": int(violations[i]),
        "first": int(first[i]), "last": int(last[i]),
        "points": [(int(x), float(values[x])) for x in touched],   # где именно цена коснулась
    }


def _analyze_window(highs, lows, closes=None):
    """Ищет фигуру в одном окне свечей. Координаты — номера свечей внутри окна."""
    n = len(highs)
    closes = closes if closes is not None else [(h + lo) / 2 for h, lo in zip(highs, lows)]
    window = _pivot_window(n)
    tol = TOUCH_TOL * float(np.mean(np.asarray(highs) - np.asarray(lows)))
    if tol <= 0:
        return None
    up = _best_boundary(pivot_highs(highs, window), highs, closes, True, tol)
    lo = _best_boundary(pivot_lows(lows, window), lows, closes, False, tol)
    if up is None or lo is None:
        return None
    if up["touches"] < MIN_PIVOTS or lo["touches"] < MIN_PIVOTS:
        return None

    x0 = min(up["first"], lo["first"])
    x1 = n - 1
    span = x1 - x0
    # касания каждой линии должны быть разнесены, а не скучены в одном углу фигуры
    if span <= 0 or min(up["last"] - up["first"], lo["last"] - lo["first"]) < MIN_SPAN * span:
        return None
    upper_at = lambda x: up["slope"] * x + up["icpt"]
    lower_at = lambda x: lo["slope"] * x + lo["icpt"]
    height = upper_at(x0) - lower_at(x0)
    gap_end = upper_at(x1) - lower_at(x1)
    if height <= 0 or gap_end < MIN_GAP_END * height:      # линии уже сошлись — фигура закончилась
        return None

    kind = classify(
        _direction(up["slope"] * span / height, FLAT_SLOPE),
        _direction(lo["slope"] * span / height, FLAT_SLOPE),
        height,
        gap_end,
    )
    if kind is None:
        return None
    # канал — две ПАРАЛЛЕЛЬНЫЕ линии, по которым цена ходит туда-обратно: на каждой нужно
    # хотя бы 2, на одной из них — 3, и почти без закрытий за линиями (иначе это просто наклон)
    if kind.startswith("channel") and (up["touches"] + lo["touches"] < CHANNEL_TOUCHES
                                       or up["violations"] + lo["violations"] > CHANNEL_VIOLATIONS):
        return None

    touches = up["touches"] + lo["touches"]
    violations = up["violations"] + lo["violations"]
    # 4 касания → 0.77, 5 → 0.85, 6 → 0.93, 7+ → 1.0; каждое нарушение −0.04
    score = float(np.clip(min(1.0, 0.45 + 0.08 * touches) - 0.04 * violations, 0, 1))
    # каждая линия рисуется от СВОЕГО первого касания — иначе она «висит» в воздухе
    # там, где цена её ещё не трогала, и фигура выглядит неточной
    xu, xl = up["first"], lo["first"]
    return {
        "kind": kind,
        "score": score,
        "touches": touches,
        "violations": violations,
        "upper": (xu, upper_at(xu), x1, upper_at(x1)),
        "lower": (xl, lower_at(xl), x1, lower_at(x1)),
        "touch_points": up["points"] + lo["points"],
        "start": x0,
        "window": n - x0,
    }


def shift(found, offset):
    """Переводит координаты фигуры из «номера свечи в окне» в «номер свечи на графике»."""
    xu, u0, x1, u1 = found["upper"]
    xl, l0, _, l1 = found["lower"]
    found.update({
        "start": found["start"] + offset,
        "upper": (xu + offset, u0, x1 + offset, u1),
        "lower": (xl + offset, l0, x1 + offset, l1),
        "touch_points": [(x + offset, y) for x, y in found["touch_points"]],
    })
    return found


def line_value(line, x):
    """Цена на линии (x0, y0, x1, y1) в свече x."""
    x0, y0, x1, y1 = line
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0


def _still_inside(found, last_close):
    upper_end = found["upper"][3]
    lower_end = found["lower"][3]
    return lower_end * (1 - BREAKOUT_TOLERANCE) <= last_close <= upper_end * (1 + BREAKOUT_TOLERANCE)


SIMILARITY_WEIGHT = 0.3     # насколько похожесть на пример пользователя влияет на выбор окна


def find_line_pattern(df, family, preferred=None, windows=WINDOWS, similarity=None, extra=None, accept=None):
    """Лучшая фигура нужного семейства, которая формируется прямо сейчас (или None).

    family — набор типов (TRIANGLES, WEDGES, CHANNELS);
    preferred — тип с рисунка пользователя: совпадение повышает оценку;
    similarity — функция (df, фигура) → похожесть на пример пользователя 0..1 или None:
    из нескольких окон выбирается фигура, больше похожая на его рисунок;
    extra — функция (df, фигура) → добавка к выбору (описание словами: подходящая фигура главнее).
    """
    highs_all = df["high"].tolist()
    lows_all = df["low"].tolist()
    closes_all = df["close"].tolist()
    last_close = float(df["close"].iloc[-1])
    best = None

    for w in windows:
        if len(df) < w:
            continue
        found = _analyze_window(highs_all[-w:], lows_all[-w:], closes_all[-w:])
        if found is None or found["kind"] not in family:
            continue
        if not _still_inside(found, last_close):
            continue
        if preferred:
            found["score"] = min(1.0, found["score"] + 0.05) if found["kind"] == preferred else found["score"] - 0.1

        shift(found, len(df) - w)         # фигура начинается с первого касания линий
        if similarity is not None:
            found["similarity"] = similarity(df, found)
        if accept is not None and not accept(found):
            continue
        # при почти равной оценке лучше длинное окно — оно захватывает фигуру целиком
        found["_rank"] = (found["score"] + 0.0005 * found["window"]
                          + SIMILARITY_WEIGHT * (found.get("similarity") or 0)
                          + (extra(df, found) if extra is not None else 0))
        if best is None or found["_rank"] > best["_rank"]:
            best = found
    if best is not None:
        best.pop("_rank")
    return best
