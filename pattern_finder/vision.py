"""Распознавание реальных скриншотов графиков.

Ничего не знает про конкретные цвета: фон определяется автоматически,
нарисованные линии ищутся как длинные прямые, свечи — как вертикальные
штрихи. Работает и в тёмной, и в светлой теме.

Главная функция: analyze_screenshot(path) -> Analysis
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

# --- настройки распознавания -------------------------------------------------
FOREGROUND_DIFF = 60        # насколько пиксель должен отличаться от фона
MIN_LINE_RATIO = 0.15       # нарисованная линия — не короче 15% ширины
MAX_LINE_GAP = 6            # разрывы в линии до 6 px считаем одной линией
VERTICAL_ANGLE = 85         # круче 85° — это свеча, а не линия
SIDE_FILL_MAX = 0.3         # по бокам нарисованной линии почти пусто
MIN_BLOB_RATIO = 0.01       # кусочки меньше 1% от главного кластера — мусор
MERGE_ANGLE = 4             # отрезки с разницей наклона < 4° ...
MERGE_DISTANCE = 8          # ... и расстоянием < 8 px — это одна линия
DOTTED_FILL = 0.75          # линия заполнена меньше чем на 75% — пунктир
CANDLE_KERNEL = 7           # свеча/фитиль — вертикальный штрих выше 7 px
CANDLE_MODE_RATIO = 0.35    # доля «вертикальных» пикселей для режима свечей
SATURATION_MIN = 60         # «цветной» пиксель (для цветных свечей)
RUN_GAP = 4                 # разрыв > 4 px в столбце отделяет свечу от объёма


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def angle(self):
        """Наклон в градусах: плюс — линия идёт вверх, минус — вниз."""
        return float(np.degrees(np.arctan2(self.y1 - self.y2, self.x2 - self.x1)))

    @property
    def length(self):
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def y_at(self, x):
        """Высота линии (в пикселях картинки) в точке x."""
        if self.x2 == self.x1:
            return float(self.y1)
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)


@dataclass
class Analysis:
    mode: str                     # "candles" или "line"
    xs: np.ndarray                # номера столбцов, где найдена цена
    prices: np.ndarray            # «цена» в пикселях (больше = выше)
    lines: list = field(default_factory=list)   # нарисованные пользователем линии
    width: int = 0
    height: int = 0
    chain: list = None            # ломаная, нарисованная пользователем (если есть)

    @property
    def ok(self):
        return len(self.prices) >= 20

    def pattern_series(self):
        """Что считать формой паттерна: нарисованную ломаную (явное намерение
        пользователя), а если её нет — цену со свечей или линии."""
        if self.chain:
            return series_from_chain(self.chain, self.height)[1]
        return self.prices


# --- загрузка ------------------------------------------------------------------
def load_image(path):
    """Открывает картинку, в том числе по пути с русскими буквами (Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть картинку: {path}")
    return normalize_size(img)


WORK_SIDE = 1280             # большие скрины (4K, ретина, «HD» в Telegram) уменьшаем до этого размера:
MAX_SIDE = 1400              # все пороги чтения (толщина рамки, ширина свечи…) подобраны для обычного размера


def normalize_size(img):
    """Огромный скрин → обычный размер. Иначе рамка прямоугольника и сглаживание в 2–3 раза
    толще, чем ждут правила, и остатки рамки читаются как сотни «свечей»."""
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return img
    k = WORK_SIDE / max(h, w)
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


# --- фон и передний план -------------------------------------------------------
def detect_background(img):
    """Фон — самый типичный цвет картинки (медиана по всем пикселям)."""
    return np.median(img.reshape(-1, 3), axis=0)


def foreground_mask(img, background):
    """255 там, где пиксель заметно отличается от фона, иначе 0."""
    diff = np.abs(img.astype(int) - background).sum(axis=2)
    return np.where(diff > FOREGROUND_DIFF, 255, 0).astype(np.uint8)


# --- нарисованные линии ---------------------------------------------------------
def _same_line(a, b):
    if abs(a.angle - b.angle) > MERGE_ANGLE:
        return False
    # перпендикулярное расстояние от середины отрезка b до прямой a
    mx = (b.x1 + b.x2) / 2
    my = (b.y1 + b.y2) / 2
    dx, dy = a.x2 - a.x1, a.y2 - a.y1
    dist = abs(dy * (mx - a.x1) - dx * (my - a.y1)) / max(a.length, 1e-9)
    return dist < MERGE_DISTANCE


def _merge(group):
    """Склеивает похожие отрезки в один — от самой левой до самой правой точки."""
    points = []
    for s in group:
        points += [(s.x1, s.y1), (s.x2, s.y2)]
    points.sort()
    (x1, y1), (x2, y2) = points[0], points[-1]
    return Line(int(x1), int(y1), int(x2), int(y2))


def _fill_ratio(mask, line):
    """Какая доля точек вдоль линии закрашена (сплошная ≈ 1, пунктир ≈ 0.5)."""
    n = max(int(line.length), 2)
    xs = np.linspace(line.x1, line.x2, n).round().astype(int)
    ys = np.linspace(line.y1, line.y2, n).round().astype(int)
    h, w = mask.shape
    xs = xs.clip(0, w - 1)
    ys = ys.clip(0, h - 1)
    # смотрим в окрестности ±1 px, чтобы не промахнуться мимо толстой линии
    hits = 0
    for x, y in zip(xs, ys):
        if mask[max(y - 1, 0):y + 2, x].any():
            hits += 1
    return hits / n


def _side_fill(mask, line, offset=6):
    """Насколько закрашено по бокам линии. Нарисованная линия тонкая — по бокам пусто.
    Цепочка свечей, случайно выстроившаяся в прямую, — толстая, по бокам тоже пиксели."""
    n = max(int(line.length), 2)
    dx = (line.x2 - line.x1) / line.length
    dy = (line.y2 - line.y1) / line.length
    nx, ny = -dy, dx                      # перпендикуляр к линии
    h, w = mask.shape
    filled = 0
    checks = 0
    for t in np.linspace(0, 1, n):
        x = line.x1 + t * (line.x2 - line.x1)
        y = line.y1 + t * (line.y2 - line.y1)
        for sign in (-1, 1):
            px = int(round(x + sign * offset * nx))
            py = int(round(y + sign * offset * ny))
            if 0 <= px < w and 0 <= py < h:
                checks += 1
                filled += mask[py, px] > 0
    return filled / checks if checks else 1.0


SHORT_LEVEL_RATIO = 0.06       # горизонтальный уровень может быть коротким — от 6% ширины


def find_lines(fg, min_ratio=MIN_LINE_RATIO, side_max=SIDE_FILL_MAX):
    """Ищет длинные прямые (нарисованные линии), склеивает дубли, убирает пунктир."""
    h, w = fg.shape
    short = min_ratio < MIN_LINE_RATIO               # короткие линии: точнее по углу, меньше голосов
    raw = cv2.HoughLinesP(
        fg, 1, np.pi / (360 if short else 180), threshold=30 if short else 60,
        minLineLength=int(w * min_ratio), maxLineGap=MAX_LINE_GAP + (2 if short else 0),
    )
    if raw is None:
        return []

    segments = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        seg = Line(int(x1), int(y1), int(x2), int(y2))
        if abs(seg.angle) <= VERTICAL_ANGLE:
            segments.append(seg)

    groups = []
    for seg in sorted(segments, key=lambda s: -s.length):
        for group in groups:
            if _same_line(group[0], seg):
                group.append(seg)
                break
        else:
            groups.append([seg])

    lines = [_merge(g) for g in groups]
    lines = [
        ln for ln in lines
        if _fill_ratio(fg, ln) >= DOTTED_FILL and _side_fill(fg, ln) <= side_max
    ]
    return sorted(lines, key=lambda ln: ln.x1)


def erase_lines(mask, lines, thickness=11):
    clean = mask.copy()
    for ln in lines:
        cv2.line(clean, (ln.x1, ln.y1), (ln.x2, ln.y2), 0, thickness)
    return clean


def erase_lines_by_color(img, mask, lines, thickness=11, tolerance=90):
    """Стирает нарисованные линии, но только пиксели ЦВЕТА линии: свеча под линией (фитиль,
    который линия задевает) остаётся целой — иначе её high/low «съедался» бы на несколько пикселей."""
    clean = mask.copy()
    h, w = mask.shape
    strokes = None
    for ln in lines:
        band = np.zeros_like(mask)
        cv2.line(band, (ln.x1, ln.y1), (ln.x2, ln.y2), 255, thickness)
        if getattr(ln, "candle_colored", False):
            if strokes is None:
                strokes = cv2.dilate(candle_color_strokes(img), np.ones((3, 3), np.uint8))
            clean[(band > 0) & (strokes > 0)] = 0
            continue
        n = max(2, int(ln.length))
        xs = np.clip(np.linspace(ln.x1, ln.x2, n).round().astype(int), 0, w - 1)
        ys = np.clip(np.linspace(ln.y1, ln.y2, n).round().astype(int), 0, h - 1)
        on = mask[ys, xs] > 0
        if not on.any():
            continue
        color = np.median(img[ys[on], xs[on]].astype(int), axis=0)
        near = np.abs(img.astype(int) - color).sum(axis=2) < tolerance
        clean[(band > 0) & near] = 0
        # тонкую серединку линии стираем всегда (антиалиасинг даёт «промежуточные» цвета)
        cv2.line(clean, (ln.x1, ln.y1), (ln.x2, ln.y2), 0, 1)
    return clean


def erase_dotted(mask):
    """Убирает тонкие горизонтальные штрихи (пунктир цены, сетку)."""
    kernel = np.ones((3, 1), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# --- свечи или линия? -------------------------------------------------------------
def candle_ratio(mask):
    """Доля пикселей, входящих в вертикальные штрихи высотой ≥ CANDLE_KERNEL."""
    total = np.count_nonzero(mask)
    if total == 0:
        return 0.0
    kernel = np.ones((CANDLE_KERNEL, 1), np.uint8)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return np.count_nonzero(vertical) / total


def keep_main_cluster(mask):
    """Оставляет основное «облако» свечей и выкидывает отдельный мусор
    (обрывки линий, кружки на концах, одинокие значки)."""
    merged = cv2.dilate(mask, np.ones((3, 15), np.uint8))   # склеиваем соседние свечи
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = areas.max()
    keep = np.zeros_like(mask)
    for i, area in enumerate(areas, start=1):
        if area >= biggest * 0.15:
            keep[labels == i] = 255
    return mask & keep


def colored_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.where(hsv[:, :, 1] >= SATURATION_MIN, 255, 0).astype(np.uint8)


# --- из маски в ряд цен -------------------------------------------------------------
def _top_run(rows):
    """Первая сверху непрерывная группа строк (свеча, а не объём под ней)."""
    breaks = np.where(np.diff(rows) > RUN_GAP)[0]
    end = breaks[0] + 1 if len(breaks) else len(rows)
    return rows[:end]


def series_from_candles(mask):
    h, w = mask.shape
    xs, prices = [], []
    for x in range(w):
        rows = np.where(mask[:, x] > 0)[0]
        if len(rows) == 0:
            continue
        run = _top_run(rows)
        xs.append(x)
        prices.append(h - 1 - (run[0] + run[-1]) / 2)   # середина свечи
    return np.array(xs), np.array(prices, dtype=float)


def remove_spikes(xs, prices, window=15, limit=4.0):
    """Убирает одиночные выбросы (столбик объёма, значок индикатора вместо свечи):
    точка, сильно отличающаяся от медианы соседей, выбрасывается."""
    if len(prices) < window:
        return xs, prices
    s = pd.Series(prices)
    med = s.rolling(window, center=True, min_periods=3).median()
    dev = (s - med).abs()
    mad = dev.rolling(window * 3, center=True, min_periods=3).median().clip(lower=1.0)
    keep = (dev <= limit * mad).to_numpy()
    return xs[keep], prices[keep]


def _intersection(a, b):
    """Точка пересечения двух прямых (или None, если параллельны)."""
    x1, y1, x2, y2 = a.x1, a.y1, a.x2, a.y2
    x3, y3, x4, y4 = b.x1, b.y1, b.x2, b.y2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def chain_from_lines(lines, gap=40):
    """Если пользователь нарисовал ломаную (отрезки идут друг за другом и
    стыкуются) — это и есть его паттерн. Возвращает вершины слева направо или None.
    Концы отрезков часто теряются за свечами, поэтому стык ищем как пересечение
    соседних прямых рядом с их концами."""
    if len(lines) < 3:
        return None
    segs = sorted(lines, key=lambda ln: ln.x1)
    points = [(segs[0].x1, segs[0].y1)]
    for a, b in zip(segs, segs[1:]):
        if (a.angle > 0) == (b.angle > 0):        # в ломаной наклон чередуется
            return None
        cross = _intersection(a, b)
        if cross is None:
            return None
        cx, cy = cross
        if not (a.x2 - gap <= cx <= b.x1 + gap):
            return None
        points.append((int(round(cx)), int(round(cy))))
    points.append((segs[-1].x2, segs[-1].y2))
    return points


def series_from_chain(points, height):
    xs = np.arange(points[0][0], points[-1][0] + 1)
    px = [p[0] for p in points]
    py = [height - 1 - p[1] for p in points]
    return xs, np.interp(xs, px, py)


def series_from_line(mask):
    h, w = mask.shape
    xs, prices = [], []
    for x in range(w):
        rows = np.where(mask[:, x] > 0)[0]
        if len(rows) == 0:
            continue
        xs.append(x)
        prices.append(h - 1 - rows.mean())
    return np.array(xs), np.array(prices, dtype=float)


# --- всё вместе ----------------------------------------------------------------------
def analyze_screenshot(path):
    img = load_image(path)
    h, w = img.shape[:2]
    background = detect_background(img)
    fg = foreground_mask(img, background)

    lines = find_lines(fg)
    without_lines = erase_dotted(erase_lines(fg, lines))

    if candle_ratio(without_lines) >= CANDLE_MODE_RATIO:
        mask = without_lines
        colored = colored_mask(img) & mask
        # цветные свечи: берём только цветное, это отсекает серый текст и оси
        if np.count_nonzero(colored) > 0.5 * np.count_nonzero(mask):
            mask = colored
        mask = keep_main_cluster(erase_dotted(mask))
        xs, prices = remove_spikes(*series_from_candles(mask))
        return Analysis("candles", xs, prices, lines, w, h, chain_from_lines(lines))

    # линейный график: «нарисованные линии» — это сама цена, их не стираем
    xs, prices = series_from_line(erase_dotted(fg))
    return Analysis("line", xs, prices, [], w, h)


# --- описание и отладочная картинка --------------------------------------------------
def describe_lines(lines):
    """Человеческое описание нарисованных линий для ответа бота."""
    if not lines:
        return "Нарисованных линий не нашёл."

    def word(angle):
        if angle > 3:
            return f"восходящая ({angle:+.0f}°)"
        if angle < -3:
            return f"нисходящая ({angle:+.0f}°)"
        return "горизонтальная"

    parts = [word(ln.angle) for ln in lines]
    text = f"Линий: {len(lines)} — " + ", ".join(parts) + "."

    if chain_from_lines(lines):
        text += " Линии образуют ломаную — ищу именно эту форму."
    elif len(lines) >= 2:
        text += " " + classify_pair(lines[0], lines[1])
    return text


def classify_pair(a, b):
    """Как ведут себя две линии: параллельны, сходятся или расходятся."""
    if abs(a.angle - b.angle) <= 5:
        return "Линии параллельны — похоже на канал."
    left = max(min(a.x1, a.x2), min(b.x1, b.x2))
    right = min(max(a.x1, a.x2), max(b.x1, b.x2))
    if right <= left:                       # линии не перекрываются по горизонтали
        left = min(a.x1, b.x1)
        right = max(a.x2, b.x2)
    gap_left = a.y_at(left) - b.y_at(left)
    gap_right = a.y_at(right) - b.y_at(right)
    if gap_left * gap_right <= 0 or abs(gap_right) < abs(gap_left):
        return "Линии сходятся — похоже на треугольник или клин."
    return "Линии расходятся — расширяющаяся формация."


def draw_debug(path, analysis, out_path):
    """Рисует поверх скриншота то, что распознал бот: цену и линии."""
    img = load_image(path)
    h = analysis.height
    if len(analysis.xs) > 1:
        pts = np.stack([analysis.xs, h - 1 - analysis.prices], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2)        # цена — жёлтым
    for ln in analysis.lines:
        cv2.line(img, (ln.x1, ln.y1), (ln.x2, ln.y2), (255, 0, 255), 3)  # линии — фиолетовым
    if analysis.chain:
        pts = np.array(analysis.chain, dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 165, 255), 4)            # ломаная — оранжевым
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(out_path)
    return out_path


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        a = analyze_screenshot(p)
        print(p, a.mode, len(a.prices), "точек", "| ломаная" if a.chain else "")
        print("  ", describe_lines(a.lines))
        draw_debug(p, a, p.rsplit(".", 1)[0] + "_debug.png")


# --- свечи со скриншота: open / high / low / close и объём ------------------------------------
GREEN_HUE = (35, 95)          # оттенок зелёного в OpenCV (0..180)
CANDLE_ZONE = 0.75            # свечи — выше 75% высоты картинки…
VOLUME_ZONE = 0.7             # …объём — столбики, которые начинаются ниже 70% высоты
BODY_SHARE = 0.6              # строка — тело свечи, если её закрашивают ≥ 60% столбцов свечи


def _runs(rows):
    """Непрерывные группы строк в столбце: [(верх, низ)]."""
    if len(rows) == 0:
        return []
    breaks = np.where(np.diff(rows) > RUN_GAP)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(rows) - 1]])
    return [(int(rows[s]), int(rows[e])) for s, e in zip(starts, ends)]


def _columns(mask, hsv, h, zone=CANDLE_ZONE):
    """По каждому столбцу картинки: верх и низ свечи, цвет (+1 зелёный, −1 красный, 0 — не понять)
    и высота столбика объёма (0 — нет)."""
    cols = []
    for x in range(mask.shape[1]):
        runs = _runs(np.where(mask[:, x] > 0)[0])
        candle = [r for r in runs if r[0] < zone * h]
        volume = [r for r in runs if r[0] >= VOLUME_ZONE * h and r[1] >= 0.9 * h]
        if not candle:
            cols.append(None)
            continue
        top, bottom = max(candle, key=lambda r: r[1] - r[0])      # самая длинная группа — свеча, не значок
        inside = mask[top:bottom + 1, x] > 0
        hues = hsv[top:bottom + 1, x, 0][inside]
        sats = hsv[top:bottom + 1, x, 1][inside]
        green = np.mean((hues >= GREEN_HUE[0]) & (hues <= GREEN_HUE[1])) if len(hues) else 0.5
        color = 1 if green > 0.6 else -1 if green < 0.4 else 0
        if len(sats) and np.mean(sats) < SATURATION_MIN:
            color = 0                                        # чёрно-белые свечи: цвета нет
        
        vol = max((r[1] - r[0] + 1 for r in volume if r[0] > bottom + RUN_GAP), default=0)
        cols.append((top, bottom, color, vol))
    return cols


def _rectangle(cols, group):
    """Широкая группа столбцов с одинаковым верхом и низом — нарисованный прямоугольник
    (ярлык с текстом, таблица индикатора, залитая зона, «позиция»). У подряд идущих свечей
    верх и низ разные."""
    tops = np.array([cols[x][0] for x in group])
    bottoms = np.array([cols[x][1] for x in group])
    return (np.mean(np.abs(tops - np.median(tops)) <= 2) >= 0.8
            and np.mean(np.abs(bottoms - np.median(bottoms)) <= 2) >= 0.8)


def _groups(cols):
    """Соседние столбцы одного цвета без пропуска — одна свеча. Слишком широкие группы
    (свечи вплотную) делим по типичной ширине свечи."""
    groups, current = [], []
    for x, c in enumerate(cols):
        if c is None or (current and cols[current[-1]][2] != c[2]):
            if current:
                groups.append(current)
            current = [x] if c is not None else []
        else:
            current.append(x)
    if current:
        groups.append(current)
    if not groups:
        return []
    width = max(1, int(np.median([len(g) for g in groups])))
    # обрывки в 1–2 столбца вплотную к соседу — та же свеча (цвет «мигает» под полупрозрачной
    # заливкой или на краю тела), а не отдельные свечи
    merged = []
    for g in groups:
        if merged and g[0] == merged[-1][-1] + 1 and min(len(g), len(merged[-1])) < 0.5 * width:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    groups = merged
    result = []
    for g in groups:
        if len(g) > 3 * max(width, 2) and _rectangle(cols, g):
            continue                       # ровный цветной прямоугольник — ярлык, таблица, зона, а не свечи
        parts = max(1, round(len(g) / width)) if len(g) > 2.5 * width else 1
        result += [list(p) for p in np.array_split(g, parts)]
    # по краям отбрасываем «острова» после большого пропуска и узкие обрывки
    # (подписи шкалы цены, значки) — пока край не станет похож на ряд свечей
    centers = [float(np.mean(g)) for g in result]
    if len(result) > 10:
        step = float(np.median(np.diff(centers)))
        width = float(np.median([len(g) for g in result]))

        with_volume = np.mean([any(cols[x][3] for x in g) for g in result]) > 0.8
        # большой пропуск у края (последние / первые 8 свечей) — всё за ним не свечи
        gaps = np.diff(centers)
        # (только если за пропуском — 1–3 «островка»: подписи, а не свечи; свечи за пропуском
        # бывают — например, бледные свечи, которые не прошли проверку цвета)
        # (или кучка обрывков уже 4 свечей — цифры шкалы цены, у них цветные края букв)
        far = [i for i in range(max(0, len(gaps) - 8), len(gaps))
               if gaps[i] > 2.2 * step and (len(gaps) - i <= 3 or centers[-1] - centers[i + 1] < 4 * step)]
        if far:
            result, centers = result[:far[0] + 1], centers[:far[0] + 1]
        gaps = np.diff(centers)
        far = [i for i in range(min(8, len(gaps)))
               if gaps[i] > 2.2 * step and (i + 1 <= 3 or centers[i] - centers[0] < 4 * step)]
        if far:
            result, centers = result[far[-1] + 1:], centers[far[-1] + 1:]

        def stray(i, neighbour):
            gap = abs(centers[i] - centers[neighbour])
            no_volume = with_volume and not any(cols[x][3] for x in result[i])   # подпись, а не свеча
            return no_volume or len(result[i]) < 0.5 * width or not 0.5 * step <= gap <= 2.2 * step

        while len(result) > 10 and stray(-1, -2):
            result, centers = result[:-1], centers[:-1]
        while len(result) > 10 and stray(0, 1):
            result, centers = result[1:], centers[1:]
    return result


def _box_lines(fg, w, h):
    """Нарисованный прямоугольник (боковик): две длинные горизонтальные линии одна над другой
    с общим отрезком по x. Возвращает (верх, низ, лево, право) в пикселях или None."""
    flat = [ln for ln in find_lines(fg) if abs(ln.angle) < 2 and ln.length > 0.25 * w]
    best = None
    for i, a in enumerate(flat):
        for b in flat[i + 1:]:
            left, right = max(a.x1, b.x1), min(a.x2, b.x2)
            gap = abs((a.y1 + a.y2) / 2 - (b.y1 + b.y2) / 2)
            if right - left < 0.7 * min(a.length, b.length) or gap < 0.15 * h:
                continue
            score = (right - left) * gap
            if best is None or score > best[0]:
                ya, yb = (a.y1 + a.y2) / 2, (b.y1 + b.y2) / 2
                best = (score, int(min(ya, yb)), int(max(ya, yb)), int(left), int(right))
    return best[1:] if best else None


def _poc_row(mask, box):
    """Линия POC внутри прямоугольника: строка с самым длинным сплошным горизонтальным отрезком."""
    top, bottom, left, right = box
    best = None
    for y in range(top + 4, bottom - 3):
        row = mask[y, left + 3:right - 3] > 0
        if not row.any():
            continue
        padded = np.concatenate([[0], row.astype(int), [0]])
        edges = np.flatnonzero(np.diff(padded))
        run = int((edges[1::2] - edges[::2]).max())
        if run >= 0.25 * (right - left) and (best is None or run > best[1]):
            best = (y, run)
    return best[0] if best else None


def _wick(full, x_range, top, bottom, gap=2):
    """Тонкие тени (фитили) часто бледные — по цвету их не видно. Достраиваем вверх и вниз
    по всем непустым пикселям центральных столбцов свечи."""
    if full is None:
        return top, bottom
    h = full.shape[0]
    column = full[:, x_range].max(axis=1) > 0
    y, miss = top - 1, 0
    while y >= 0 and miss <= gap:
        miss = 0 if column[y] else miss + 1
        y -= 1
    new_top = y + 1 + miss
    y, miss = bottom + 1, 0
    while y < h and miss <= gap:
        miss = 0 if column[y] else miss + 1
        y += 1
    new_bottom = y - 1 - miss
    return min(top, new_top), max(bottom, new_bottom)


def _drop_strays(rows, window=4, far=6.0):
    """График непрерывный: каждая свеча рядом по цене с соседями. Обрывки подписей сверху
    (тикер, цены), шкалы времени и логотипа снизу стоят далеко от соседей и маленькие — убираем.
    Большую свечу (импульс) не трогаем, даже если она далеко."""
    if len(rows) < 2 * window + 3:
        return rows
    sizes = np.array([r["high"] - r["low"] for r in rows])
    typical = max(float(np.median(sizes)), 1.0)
    # 6+ «свечей» подряд с ОДИНАКОВЫМ верхом и низом и крупнее обычной — это залитая зона / плашка
    flat = np.zeros(len(rows), bool)
    start = 0
    for i in range(1, len(rows) + 1):
        same = i < len(rows) and abs(rows[i]["high"] - rows[start]["high"]) <= 2 \
            and abs(rows[i]["low"] - rows[start]["low"]) <= 2
        if not same:
            if i - start >= 6 and sizes[start] >= 2 * typical:
                flat[start:i] = True
            start = i
    rows = [r for r, f in zip(rows, flat) if not f]
    if len(rows) < 2 * window + 3:
        return rows
    mids = np.array([(r["high"] + r["low"]) / 2 for r in rows])
    sizes = np.array([r["high"] - r["low"] for r in rows])
    keep = []
    for i, r in enumerate(rows):
        near = np.r_[mids[max(0, i - window):i], mids[i + 1:i + 1 + window]]
        center = float(np.median(near))
        if abs(mids[i] - center) > far * typical and sizes[i] < 2 * typical:
            continue
        keep.append(r)
    return keep


def _same_candle(parts):
    """Столбцы одной свечи перекрываются по высоте. Если в каком-то столбце «самой длинной группой»
    оказался текст подписи или значок далеко сверху / снизу — он не часть свечи: иначе свеча
    растянулась бы на всю картинку. Оставляем столбцы, связанные по высоте со средним."""
    if len(parts) <= 1:
        return parts
    middle = parts[len(parts) // 2]
    top, bottom = middle[0], middle[1]
    keep = [middle]
    changed = True
    rest = [p for i, p in enumerate(parts) if i != len(parts) // 2]
    while changed:
        changed = False
        for p in list(rest):
            if p[0] <= bottom + 2 and p[1] >= top - 2:
                keep.append(p)
                rest.remove(p)
                top, bottom = min(top, p[0]), max(bottom, p[1])
                changed = True
    return keep


def _candle_rows(img, mask, h, zone=CANDLE_ZONE, full=None):
    cols = _columns(mask, cv2.cvtColor(img, cv2.COLOR_BGR2HSV), h, zone)
    rows = []
    for g in _groups(cols):
        parts = _same_candle([cols[x] for x in g])
        top = min(p[0] for p in parts)
        bottom = max(p[1] for p in parts)
        center = int(round(np.mean(g)))
        top, bottom = _wick(full, slice(max(0, center - 1), center + 2), top, bottom)
        cover = np.zeros(bottom - top + 1)
        for p in parts:
            cover[p[0] - top:p[1] - top + 1] += 1
        if len(g) >= 3:
            body = np.where(cover >= BODY_SHARE * len(g))[0]
            body_top, body_bottom = (top + body[0], top + body[-1]) if len(body) else (top, bottom)
        else:                                             # тонкая свеча: тело не отличить от тени
            quarter = (bottom - top) / 4
            body_top, body_bottom = top + quarter, bottom - quarter
        color = np.sign(sum(p[2] for p in parts))
        rows.append({"x": float(np.mean(g)), "width": len(g), "high": float(h - 1 - top), "low": float(h - 1 - bottom),
                     "upper": float(h - 1 - body_top), "lower": float(h - 1 - body_bottom),
                     "color": int(color), "volume": float(max(p[3] for p in parts))})
    return _drop_strays(rows)


def _to_frame(rows, h):
    df = pd.DataFrame(rows)
    # выбросы: серые столбы-метки на всю высоту (курсор). Цветная длинная свеча — это импульс, её оставляем
    tall = df["high"] - df["low"]
    df = df[~(((tall >= 0.6 * h) & (df["color"] == 0)) | (tall >= 0.95 * h))].reset_index(drop=True)
    # направление: по цвету; если цвета нет (чёрно-белый график) — по следующей свече
    nxt = df["close"] if "close" in df else ((df["upper"] + df["lower"]) / 2).shift(-1)
    up = (df["color"] > 0) | ((df["color"] == 0) & (nxt.fillna((df["upper"] + df["lower"]) / 2)
                                                     > (df["upper"] + df["lower"]) / 2))
    df["open"] = np.where(up, df["lower"], df["upper"])
    df["close"] = np.where(up, df["upper"], df["lower"])
    if (df["volume"] > 0).mean() < 0.5:
        df = df.drop(columns="volume")                     # объёма на скрине нет
    df["open_time"] = pd.date_range("2000-01-01", periods=len(df), freq="h", tz="UTC")
    return df


FILL_DIFF = 12               # заливка прямоугольника — бледная, но отличается от фона
FILL_SOLID = 0.85            # прямоугольник залит сплошь (≥ 85% площади рамки)
FILL_COLOR = 45              # пиксели цвета заливки (± 45) стираем, свечи под ней — нет


def filled_boxes(img, background):
    """Залитые прямоугольники, которые ты нарисовал на скрине (зона OB, FVG, боковик) —
    и зона, которую рисует сам бот на «примере с рынка». Возвращает [(лево, верх, право, низ, цвет)].
    Свечи тонкие и стоят через промежутки — большой «сплошной квадрат» они не дают.
    Сначала ищем обычные зоны; не нашли — ищем ТОНКУЮ зону (FVG / OB высотой в пару свечей)."""
    h, w = img.shape[:2]
    boxes = _filled_boxes(img, background, max(9, int(min(h, w) * 0.04)), 0.04)
    if not boxes:
        boxes = _filled_boxes(img, background, max(4, int(min(h, w) * 0.008)), 0.012, strict=True)
    return boxes


def _filled_boxes(img, background, k, min_height, strict=False):
    h, w = img.shape[:2]
    diff = np.abs(img.astype(int) - background).sum(axis=2)
    soft = (diff > FILL_DIFF).astype(np.uint8)
    solid = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if strict and bw >= 0.08 * w:
            # к тонкой зоне прилипают свечи сверху и снизу — рамку зоны берём по строкам, где почти
            # вся ширина — цвет заливки
            part = img[y:y + bh, x:x + bw].astype(int)
            fill = np.median(part[labels[y:y + bh, x:x + bw] == i], axis=0)
            if np.abs(fill - background).sum() > 3 * FOREGROUND_DIFF:
                continue                                       # не бледная заливка (тело свечи и т.п.)
            near = (np.abs(part - fill).sum(axis=2) < FILL_COLOR) & (diff[y:y + bh, x:x + bw] > FILL_DIFF)
            rows = np.flatnonzero(near.mean(axis=1) >= 0.6)
            if len(rows) == 0:
                continue
            y, bh = y + int(rows[0]), int(rows[-1] - rows[0] + 1)
            cols = np.flatnonzero(near[rows[0]:rows[-1] + 1].mean(axis=0) >= 0.6)
            if len(cols) == 0:
                continue
            x, bw = x + int(cols[0]), int(cols[-1] - cols[0] + 1)
            area = int(near[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].sum())
        # у тонкой зоны строки рамки и сетки внутри не цвета заливки — допускаем меньшую «сплошность»
        if bw < 0.08 * w or bh < min_height * h or area < (0.6 if strict else FILL_SOLID) * bw * bh:
            continue
        if strict and bw < 3 * bh:
            continue                                           # тонкая зона — вытянута по горизонтали
        if strict:
            region = img[y:y + bh, x:x + bw].reshape(-1, 3)
            boxes.append((int(x), int(y), int(x + bw - 1), int(y + bh - 1), np.median(region, axis=0)))
            continue
        region = img[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == i]
        color = np.median(region, axis=0)
        if np.abs(color - background).sum() > 3 * FOREGROUND_DIFF:
            continue                                           # не бледная заливка, а что-то тёмное / яркое
        same = np.abs(region.astype(int) - color).sum(axis=1) < FILL_COLOR
        if np.mean(same) < (0.9 if strict else 0.7):
            continue                                           # не однотонно — это не заливка
        boxes.append((int(x), int(y), int(x + bw - 1), int(y + bh - 1), color))
    return boxes


def erase_boxes(img, boxes, background):
    """Стирает заливку и рамку прямоугольников (до цвета фона), свечи под ними остаются."""
    clean = img.copy()
    h, w = img.shape[:2]
    for left, top, right, bottom, color in boxes:
        y0, y1, x0, x1 = max(0, top - 4), min(h, bottom + 5), max(0, left - 4), min(w, right + 5)
        part = clean[y0:y1, x0:x1].astype(int)
        near_fill = np.abs(part - color).sum(axis=2) < FILL_COLOR
        # рамка: полоса у краёв — её цвет похож на заливку оттенком, но темнее
        band = np.zeros(near_fill.shape, bool)
        band[:9, :] = band[-9:, :] = True
        band[:, :9] = band[:, -9:] = True
        hsv = cv2.cvtColor(clean[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(int)
        fill_hue = int(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        hue_gap = np.minimum(np.abs(hsv[..., 0] - fill_hue), 180 - np.abs(hsv[..., 0] - fill_hue))
        frame = band & (hue_gap < 12) & (np.abs(part - background).sum(axis=2) > FILL_DIFF)
        clean[y0:y1, x0:x1][near_fill | frame] = background.astype(np.uint8)
        _erase_outline(clean, img, (left, top, right, bottom), background)
    return clean


def _erase_outline(clean, img, box, background, reach=14):
    """Рамка прямоугольника ЛЮБОГО цвета (красная, голубая…): тонкие линии по краям зоны
    (± несколько пикселей от края заливки). Строки и столбцы рамки — те, где почти по всей длине
    стороны есть «не фон». Вертикальные стороны не трогаем там, где столбец уходит за пределы
    зоны — это свеча, которая стоит прямо на краю."""
    left, top, right, bottom = box
    h, w = img.shape[:2]
    img = clean.copy()                       # заливку уже стёрли — она не мешает искать рамку
    differs = np.abs(img.astype(int) - background).sum(axis=2) > FILL_DIFF * 2
    inner_x = slice(left + 5, max(left + 6, right - 5))
    inner_y = slice(top + 5, max(top + 6, bottom - 5))
    width, height = max(1, right - left - 10), max(1, bottom - top - 10)
    rows = [y for y in list(range(top - reach, top + reach + 1)) + list(range(bottom - reach, bottom + reach + 1))
            if 0 <= y < h and differs[y, inner_x].sum() >= 0.6 * width]
    cols = [x for x in list(range(left - reach, left + reach + 1)) + list(range(right - reach, right + reach + 1))
            if 0 <= x < w and differs[inner_y, x].sum() >= 0.6 * height]
    if not rows:
        return
    y0, y1 = min(rows), max(rows)
    xs = slice(max(0, left - reach), min(w, right + reach + 1))
    # цвет рамки — по середине горизонтальных сторон (там нет свечей)
    color = np.median(np.concatenate([img[y, inner_x].reshape(-1, 3) for y in rows]).astype(int), axis=0)
    for a, b in _clusters(rows, gap=2):
        a, b = max(0, a - 1), min(h - 1, b + 1)        # с краями сглаживания
        # свеча, которая проходит СКВОЗЬ сторону рамки (есть «не фон» и над, и под полосой), — не трогаем
        crossing = differs[max(0, a - 1), xs] & differs[min(h - 1, b + 1), xs]
        for y in range(a, b + 1):
            row = clean[y, xs]
            row[~crossing] = background.astype(np.uint8)
    # вертикальные стороны: стираем, но только там, где «не фон» по горизонтали узкий (сама линия
    # рамки); если рядом широкое тело свечи — это свеча, её не трогаем
    for c0, c1 in _clusters(cols, gap=2):
        c0, c1 = max(0, c0 - 1), min(w - 1, c1 + 1)
        line = c1 - c0 + 1
        # свеча стоит ровно на стороне рамки и продолжается за её пределы (выше или ниже) — не трогаем
        beyond = lambda rows: max((differs[rows, x].sum() for x in range(c0, c1 + 1)), default=0)
        # свеча стоит ровно на стороне рамки и продолжается за её пределы — стираем только
        # пиксели цвета рамки (свеча другого цвета остаётся целой)
        candle_on_side = (beyond(slice(max(0, y0 - 9), max(0, y0 - 1))) >= 5
                          or beyond(slice(min(h, y1 + 2), min(h, y1 + 10))) >= 5)
        for y in range(y0, y1 + 1):
            if candle_on_side:
                part = clean[y, c0:c1 + 1]
                part[np.abs(img[y, c0:c1 + 1].astype(int) - color).sum(axis=-1) < 110] = background.astype(np.uint8)
                continue
            # сплошной «не фон» вокруг рамки в этой строке: если он намного шире самой линии —
            # рядом тело свечи, строку не трогаем
            lo = c0
            while lo > 0 and differs[y, lo - 1] and c0 - lo < 12:
                lo -= 1
            hi = c1
            while hi < w - 1 and differs[y, hi + 1] and hi - c1 < 12:
                hi += 1
            if (hi - lo + 1) - line <= 2:
                clean[y, c0:c1 + 1] = background.astype(np.uint8)
            else:                                        # рядом свеча: стираем только цвет рамки
                part = clean[y, c0:c1 + 1]
                part[np.abs(img[y, c0:c1 + 1].astype(int) - color).sum(axis=-1) < 110] = background.astype(np.uint8)


def _long_runs(mask, axis, share):
    """Строки (axis=1) или столбцы (axis=0), где есть сплошной отрезок ≥ share длины."""
    data = mask > 0 if axis == 1 else (mask > 0).T
    size = data.shape[1]
    result = []
    for i, row in enumerate(data):
        if row.sum() < share * size:
            continue
        padded = np.concatenate([[0], row.astype(int), [0]])
        edges = np.flatnonzero(np.diff(padded))
        if (edges[1::2] - edges[::2]).max() >= share * size:
            result.append(i)
    return result


def _clusters(values, gap=3):
    groups = []
    for v in values:
        if groups and v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(g[0], g[-1]) for g in groups]


def chart_frames(fg):
    """Рамки графиков (как у matplotlib: свечи сверху, объём снизу — две рамки одна под другой).
    Возвращает [(верх, низ, лево, право)] — внутренности рамок."""
    h, w = fg.shape
    lines = _clusters(_long_runs(fg, 1, 0.6))
    frames = []
    for (a0, a1), (b0, b1) in zip(lines, lines[1:]):
        top, bottom = a1 + 1, b0 - 1
        if bottom - top < 0.15 * h:
            continue
        cols = _clusters(_long_runs(fg[top:bottom + 1], 0, 0.9))
        if len(cols) < 2 or cols[-1][0] - cols[0][1] < 0.6 * w:
            continue
        frames.append((top + 2, bottom - 2, cols[0][1] + 3, cols[-1][0] - 3))
    return frames


def main_panel(img, fg):
    """Если это картинка из нескольких графиков в рамках (например, «пример с рынка» от бота:
    свечи + объём + подписи осей) — оставляем самый большой график. Иначе — None."""
    frames = chart_frames(fg)
    if len(frames) < 2:
        return None
    lefts = [f[2] for f in frames]
    if max(lefts) - min(lefts) > 0.05 * fg.shape[1]:
        return None                                          # рамки не одна под другой — не графики
    return max(frames, key=lambda f: f[1] - f[0])


def drawn_color_lines(img, min_ratio=0.04):
    """Прямые цвета рисунка: насыщенный цвет, но не зелёный и не красный (это цвета свечей)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    candle = ((hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])) | (hue <= 12) | (hue >= 160)
    background = detect_background(img)
    visible = (np.abs(img.astype(int) - background).sum(axis=2) > FOREGROUND_DIFF) & (hsv[..., 2] >= 80)
    mask = np.where((sat >= 90) & ~candle & visible, 255, 0).astype(np.uint8)
    strokes = candle_color_strokes(img)               # линии красного / зелёного цвета (как свечи)
    plain = mask.copy()
    mask = cv2.bitwise_or(mask, strokes)
    if np.count_nonzero(mask) < 20:
        return []
    h, w = mask.shape
    raw = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=20, minLineLength=int(w * min_ratio), maxLineGap=4)
    if raw is None:
        return []
    segments = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        seg = Line(int(x1), int(y1), int(x2), int(y2))
        if abs(seg.angle) <= 60:
            segments.append(seg)
    groups = []
    for seg in sorted(segments, key=lambda s: -s.length):
        for group in groups:
            if _same_line(group[0], seg):
                group.append(seg)
                break
        else:
            groups.append([seg])
    lines = [_merge(g) for g in groups]
    for ln in lines:                                  # линия цвета свечи — стирать только её штрих
        n = max(2, int(ln.length))
        xs = np.clip(np.linspace(ln.x1, ln.x2, n).round().astype(int), 0, w - 1)
        ys = np.clip(np.linspace(ln.y1, ln.y2, n).round().astype(int), 0, h - 1)
        near = cv2.dilate(plain, np.ones((5, 5), np.uint8))[ys, xs] > 0
        ln.candle_colored = bool(near.mean() < 0.5)
    return lines


STROKE_LEN = 0.035           # штрих цвета свечи — линия, только если длиннее 3.5% ширины скрина


def candle_color_strokes(img):
    """Пиксели красного / зелёного цвета, которые НЕ часть свечей. Свеча — вертикальный столбик
    (тело и фитиль), а нарисованная линия — длинный тонкий горизонтальный / наклонный штрих.
    Убираем всё вертикальное — остаются штрихи, нарисованные человеком."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    candle = ((hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])) | (hue <= 12) | (hue >= 160)
    background = detect_background(img)
    visible = (np.abs(img.astype(int) - background).sum(axis=2) > FOREGROUND_DIFF) & (hsv[..., 2] >= 80)
    colored = np.where((sat >= 50) & candle & visible, 255, 0).astype(np.uint8)
    if np.count_nonzero(colored) < 20:
        return colored
    vertical = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8))
    vertical = cv2.dilate(vertical, np.ones((3, 3), np.uint8))
    rest = cv2.bitwise_and(colored, cv2.bitwise_not(vertical))
    rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # сглаживание рвёт штрих
    # тела свечей — широкие прямоугольники: их верх/низ тоже горизонтальны, но короткие (ширина свечи);
    # оставляем только длинные связные штрихи
    count, labels, stats, _ = cv2.connectedComponentsWithStats(rest)
    keep = np.zeros_like(rest)
    min_len = STROKE_LEN * img.shape[1]
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if max(bw, bh) >= min_len and area <= 4 * max(bw, bh):     # длинный и тонкий
            keep[labels == i] = 255
    return keep


def _grid_line(img, line):
    """Линия сетки графика: горизонталь серого цвета почти на всю ширину картинки."""
    h, w = img.shape[:2]
    at_edge = line.x1 <= 0.02 * w or line.x2 >= 0.98 * w        # сетка идёт от края картинки
    if abs(line.angle) >= 1 or not (at_edge or line.x2 - line.x1 >= 0.85 * w):
        return False
    y = int(np.clip(round((line.y1 + line.y2) / 2), 0, h - 1))
    xs = np.linspace(line.x1, line.x2, 50).round().astype(int).clip(0, w - 1)
    hsv = cv2.cvtColor(img[y, xs][None], cv2.COLOR_BGR2HSV)[0]
    return float(np.median(hsv[:, 1])) < 40


def _candle_like(img, line):
    """Ряд мелких зелёных и красных свечей тоже похож на прямую — такую «линию» не стираем."""
    n = max(2, int(line.length))
    xs = np.linspace(line.x1, line.x2, n).round().astype(int)
    ys = np.linspace(line.y1, line.y2, n).round().astype(int)
    ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
    pixels = img[ys[ok], xs[ok]]
    # линию, нарисованную человеком, видно СПЛОШЬ по всей длине (даже красную или зелёную поверх
    # свечей); у ряда мелких свечей между ними — фон
    background = detect_background(img)
    if len(pixels) and np.mean(np.abs(pixels.astype(int) - background).sum(axis=1) > FOREGROUND_DIFF) >= 0.8:
        return False
    hsv = cv2.cvtColor(pixels[None], cv2.COLOR_BGR2HSV)[0].astype(int)
    colored = hsv[hsv[:, 1] >= SATURATION_MIN]
    if len(colored) < 0.3 * n:
        return False
    green = np.mean((colored[:, 0] >= GREEN_HUE[0]) & (colored[:, 0] <= GREEN_HUE[1]))
    red = np.mean((colored[:, 0] <= 10) | (colored[:, 0] >= 160))
    return green >= 0.15 and red >= 0.15


def erase_labels(img, background, protect=()):
    """Цветные ярлыки с текстом, таблички индикаторов, «позиции» и непрозрачные зоны — плоские
    полосы ОДНОГО цвета: в каждой строке длинный сплошной отрезок, и так несколько строк подряд.
    Свечи так не выглядят (они узкие, а соседние — разного цвета и высоты). Стираем такие полосы
    (то, что под непрозрачным ярлыком, и так не видно). Тонкие линии (уровни BOS, EQH) не трогаем."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    diff = np.abs(img.astype(int) - background).sum(axis=2)
    colored = (hsv[..., 1] >= SATURATION_MIN) & (diff > FOREGROUND_DIFF)
    if colored.sum() < 50:
        return img
    hue = hsv[..., 0].astype(int)
    classes = [(hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]), (hue <= 12) | (hue >= 160)]
    classes.append(~(classes[0] | classes[1]))
    length = int(max(40, 0.04 * w))
    found = np.zeros((h, w), np.uint8)
    # серая табличка индикатора (тёмная или светлая плашка с цветным текстом)
    plate = (hsv[..., 1] < 40) & (diff > 150)             # плотная плашка, а не бледная заливка зоны
    plate = cv2.morphologyEx(plate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((12, length), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(plate)
    tables = np.zeros((h, w), np.uint8)
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area >= 0.8 * bw * bh and bh <= 0.35 * h and bw <= 0.5 * w:
            tables[y:y + bh, x:x + bw] = 1
    for cls in classes:
        mask = (colored & cls).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))       # буквы внутри ярлыка
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))   # только длинные отрезки
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 1), np.uint8))        # и толстые полосы
        found |= mask
    if not found.any() and not tables.any():
        return img
    clean = img.copy()
    clean[tables > 0] = background.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(found)
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        if bh > 0.12 * h:
            continue                                   # высокий блок — не ярлык (вдруг это свечи)
        if y + bh >= 0.9 * h:
            continue                                   # у нижнего края — столбики объёма, а не ярлык
        inside = any(min(x + bw, r + 1) - max(x, l) > 0.5 * bw and min(y + bh, b + 1) - max(y, t) > 0.5 * bh
                     for l, t, r, b, _ in protect)
        if inside:
            continue                                   # это залитая зона пользователя — её не стираем
        clean[max(0, y - 1):y + bh + 1, max(0, x - 1):x + bw + 1] = background.astype(np.uint8)
    return clean


def erase_legend(img, background):
    """Подписи сверху слева (тикер, O H L C зелёными и красными цифрами, индикаторы) — строки
    мелкого текста. Текст: много маленьких значков на одной линии вплотную друг к другу.
    Свечи так не стоят (они разной высоты и с ровным шагом). Закрашиваем такие строки фоном."""
    h, w = img.shape[:2]
    zone = int(0.2 * h)
    if zone < 20:
        return img
    diff = np.abs(img[:zone].astype(int) - background).sum(axis=2)
    fg = (diff > FOREGROUND_DIFF).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    small = [i for i in range(1, count) if stats[i][3] <= max(14, 0.03 * h) and stats[i][2] <= 0.05 * w]
    rows = {}
    for i in small:
        x, y, bw, bh, _ = stats[i]
        rows.setdefault(int((y + bh / 2) // 6), []).append(i)
    clean = img.copy()
    for band, items in rows.items():
        items = sorted(items, key=lambda i: stats[i][0])
        run = [items[0]]
        for i in items[1:] + [None]:
            if i is not None and stats[i][0] - (stats[run[-1]][0] + stats[run[-1]][2]) <= 12:
                run.append(i)
                continue
            heights = [stats[k][3] for k in run]
            if len(run) >= 6 and max(heights) - min(heights) <= 10:       # строка текста
                x0 = min(stats[k][0] for k in run)
                x1 = max(stats[k][0] + stats[k][2] for k in run)
                y0 = min(stats[k][1] for k in run)
                y1 = max(stats[k][1] + stats[k][3] for k in run)
                for k in run:
                    clean[:zone][labels == k] = background.astype(np.uint8)
            run = [i] if i is not None else []
    return clean


def _prepare(path, with_boxes=False):
    img = load_image(path)
    background = detect_background(img)
    img = erase_legend(img, background)                   # подписи TradingView сверху слева
    img = erase_labels(img, background, filled_boxes(img, background))
    panel = main_panel(img, foreground_mask(img, background))
    if panel is not None:                        # несколько графиков в рамках — берём главный
        top, bottom, left, right = panel
        img = img[top:bottom + 1, left:right + 1].copy()
    h, w = img.shape[:2]
    boxes = filled_boxes(img, background)
    if boxes:                                    # заливку не путаем со свечами
        img = erase_boxes(img, boxes, background)
    fg = foreground_mask(img, background)
    lines = [ln for ln in find_lines(fg) if not _candle_like(img, ln) and not _grid_line(img, ln)]
    # короткие ГОРИЗОНТАЛЬНЫЕ линии (уровень BOS, EQH/EQL) — трейдер часто рисует их коротко
    short = [ln for ln in find_lines(fg, SHORT_LEVEL_RATIO, side_max=0.6)
             if not _candle_like(img, ln) and not _grid_line(img, ln)
             and not any(_same_line(ln, other) for other in lines)]
    lines += [ln for ln in short if abs(ln.angle) < 2]
    # линии «цвета рисунка» (не зелёные и не красные, как свечи) — точно нарисованы тобой, даже короткие
    for ln in drawn_color_lines(img):
        if not any(_same_line(ln, other) for other in lines):
            lines.append(ln)
    # короткие НАКЛОННЫЕ — только парой (две границы флага / вымпела рядом друг с другом)
    sloped = [ln for ln in short if 2 <= abs(ln.angle) <= 60]
    for i, a in enumerate(sloped):
        for b in sloped[i + 1:]:
            overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
            if overlap >= 0.6 * min(a.x2 - a.x1, b.x2 - b.x1) and abs(a.angle - b.angle) < 40 \
                    and abs((a.y1 + a.y2) - (b.y1 + b.y2)) / 2 > 6:
                lines += [ln for ln in (a, b) if ln not in lines]
    mask = erase_dotted(erase_lines_by_color(img, fg, lines))
    PANEL["cropped"] = panel is not None
    PANEL["lines"] = lines
    if with_boxes:
        return img, h, w, fg, mask, boxes
    return img, h, w, fg, mask


PANEL = {"cropped": False, "lines": []}   # вырезали ли главный график из рамки; нарисованные линии


def _has_volume(mask, h):
    """Внизу скрина есть объём: много столбиков, которые стоят на ОДНОЙ линии-основании
    в нижней трети. У свечей низ разный — если основания нет, внизу просто свечи
    (цена ушла вниз), и их нельзя отрезать как «объём»."""
    bottoms = []
    for x in range(mask.shape[1]):
        runs = _runs(np.where(mask[:, x] > 0)[0])
        low = [r for r in runs if r[0] >= VOLUME_ZONE * h]
        if low:
            bottoms.append(low[-1][1])
    if len(bottoms) < 0.08 * mask.shape[1]:
        return False
    counts = np.bincount(bottoms)
    base = int(np.argmax(counts))
    same = sum(1 for b in bottoms if abs(b - base) <= 1)
    return same >= 0.6 * len(bottoms) and same >= 0.08 * mask.shape[1]


def _zone(mask=None, h=None):
    """Где кончаются свечи: внизу обычного скрина бывает объём, в вырезанной рамке — нет.
    Объёма не видно — свечи могут быть до самого низа картинки."""
    if PANEL["cropped"]:
        return 1.01
    if mask is not None and h is not None and not _has_volume(mask, h):
        return 1.01
    return CANDLE_ZONE


def _colored(img, mask):
    colored = colored_mask(img) & mask
    return colored if np.count_nonzero(colored) > 0.5 * np.count_nonzero(mask) else mask


def extract_candles(path):
    """Свечи с картинки в пикселях (цена вверх = больше): DataFrame open/high/low/close
    (+ volume, если внизу есть столбики объёма) или None, если свечей не видно.
    Масштаб — пиксели, но для правил SMC это не важно: всё меряется в средних свечах (ATR)."""
    return extract_candles_and_zones(path)[0]


MIN_CANDLES = 12            # крупный скрин: 12 свечей уже достаточно


def _drop_box_sides(rows, boxes, h):
    """Остаток боковой стороны рамки, похожий на свечу: узкий «столбик» ровно у левого или правого
    края зоны на всю её высоту. Настоящая свеча (даже OB, совпадающий с зоной) — шире."""
    if not rows or not boxes:
        return rows
    typical = float(np.median([r["width"] for r in rows]))
    keep = []
    for r in rows:
        fake = False
        for left, top, right, bottom, _ in boxes:
            at_side = min(abs(r["x"] - left), abs(r["x"] - right)) <= max(4, typical)
            span = (r["high"] - r["low"]) >= 0.8 * (bottom - top)
            reach = max(12, 3 * typical)          # к рамке может прилипнуть маленькая свеча рядом
            inside = r["high"] <= h - 1 - top + reach and r["low"] >= h - 1 - bottom - reach
            if at_side and span and inside and r["width"] < 0.6 * typical:
                fake = True
        if not fake:
            keep.append(r)
    return keep


def _wick_mask(mask, h):
    """Маска для фитилей: все непустые пиксели (нарисованные линии уже стёрты). Объём снизу
    отделён от свечи промежутком — достраивание фитиля на нём останавливается."""
    return mask


def _zones_on_candles(df, boxes, h):
    """Залитые прямоугольники → зоны на свечах: с какой свечи начинается, верх и низ (в «пикселях вверх»)."""
    zones = []
    xs = df["x"].to_numpy()
    for left, top, right, bottom, _ in boxes:
        inside = np.flatnonzero(xs >= left - 2)               # первая свеча у левого края зоны
        start = int(inside[0]) if len(inside) else len(xs) - 1
        end = int(np.argmin(np.abs(xs - right)))
        zones.append({"start": start, "end": max(start, end),
                      "top": float(h - 1 - top), "bottom": float(h - 1 - bottom)})
    return zones


def _lines_on_candles(df, lines, h):
    """Нарисованные линии → в координатах свечей: x — номер свечи (дробный), y — «пиксели вверх»."""
    xs = df["x"].to_numpy()
    index = np.arange(len(xs), dtype=float)
    result = []
    for ln in lines:
        x0, x1 = np.interp([ln.x1, ln.x2], xs, index)
        result.append({"x0": float(x0), "x1": float(x1), "y0": float(h - 1 - ln.y1), "y1": float(h - 1 - ln.y2),
                       "flat": abs(ln.angle) < 3})
    return result


def extract_marks(path):
    """Свечи и всё, что ты нарисовал на скрине, в координатах свечей:
    (свечи, закрашенные зоны, линии) — или (None, [], []), если свечей не видно."""
    img, h, w, fg, mask, boxes = _prepare(path, with_boxes=True)
    lines = PANEL["lines"]
    if candle_ratio(mask) < CANDLE_MODE_RATIO:
        return None, [], []                                # линейный график — свечей нет
    rows = _drop_box_sides(_candle_rows(img, _colored(img, mask), h, _zone(mask, h), full=_wick_mask(mask, h)), boxes, h)
    if len(rows) < MIN_CANDLES:
        return None, [], []
    df = _to_frame(rows, h)
    zones = _zones_on_candles(df, boxes, h)
    marks = _lines_on_candles(df, lines, h)
    for extra in _zones_from_lines(marks, h):     # линии-границы зоны помечены "zone": True
        same = any(abs(extra["start"] - z["start"]) <= 2 and min(extra["top"], z["top"]) - max(extra["bottom"], z["bottom"])
                   > 0.5 * (extra["top"] - extra["bottom"]) for z in zones)
        if not same:                              # рамка уже найденной залитой зоны — не вторая зона
            zones.append(extra)
    cols = ["open_time", "open", "high", "low", "close"] + (["volume"] if "volume" in df else [])
    return df[cols], zones, marks


def _zones_from_lines(marks, h):
    """Узкая зона, у которой заливка не видна, — это две горизонтальные линии одна над другой
    с общим началом и концом (верх и низ прямоугольника)."""
    flat = [m for m in marks if m["flat"]]
    zones = []
    for i, a in enumerate(flat):
        for b in flat[i + 1:]:
            if a.get("zone") or b.get("zone"):
                continue
            gap = abs(a["y0"] - b["y0"])
            if abs(a["x0"] - b["x0"]) <= 1.5 and abs(a["x1"] - b["x1"]) <= 2 and 3 <= gap <= 0.2 * h:
                a["zone"] = b["zone"] = True
                zones.append({"start": int(round(min(a["x0"], b["x0"]))), "end": int(round(max(a["x1"], b["x1"]))),
                              "top": max(a["y0"], b["y0"]), "bottom": min(a["y0"], b["y0"])})
    return zones


def extract_candles_and_zones(path):
    """То же, что extract_candles, плюс зоны, которые ты закрасил на скрине: (свечи, [зоны])."""
    df, zones, _ = extract_marks(path)
    return df, zones


def extract_range(path):
    """Скрин боковика, где ты НАРИСОВАЛ прямоугольник (и, может быть, профиль объёма с линией POC):
    (свечи, боковик) или (свечи, None), если прямоугольника нет. Боковик — номера первой и последней
    свечи внутри прямоугольника, верх и низ (в тех же «пикселях вверх», что и свечи) и POC."""
    img, h, w, fg, mask, boxes = _prepare(path, with_boxes=True)
    box = _box_lines(fg, w, h)
    if box is None and boxes:                    # прямоугольник с заливкой: рамку стёрли вместе с ней
        left, top, right, bottom, _ = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        box = (top, bottom, left, right)
    if box is None:
        return extract_candles(path), None
    top, bottom, left, right = box
    poc_y = _poc_row(fg, box)
    # профиль объёма и линия POC — горизонтальные полосы: стираем их, чтобы не спутать со свечами
    wide = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 25), np.uint8))
    clean = mask.copy()
    clean[top:bottom + 1, left:right + 1] &= ~wide[top:bottom + 1, left:right + 1]
    clean[max(0, top - 2):top + 3, left:right + 1] = 0                 # рамку прямоугольника — тоже
    clean[bottom - 2:bottom + 3, left:right + 1] = 0
    clean[top:bottom + 1, max(0, left - 2):left + 3] = 0                # и боковые стороны рамки
    clean[top:bottom + 1, max(0, right - 2):right + 3] = 0
    if candle_ratio(clean) < CANDLE_MODE_RATIO * 0.7:
        return None, None
    rows = _candle_rows(img, _colored(img, clean), h, _zone(clean, h), full=clean)
    if len(rows) < MIN_CANDLES:
        return None, None
    df = _to_frame(rows, h)
    inside = np.flatnonzero((df["x"] >= left - 2) & (df["x"] <= right + 2))
    if len(inside) < 10:
        return df, None
    found = {"start": int(inside[0]), "end": int(inside[-1]),
             "top": float(h - 1 - top), "bottom": float(h - 1 - bottom),
             "poc": None if poc_y is None else float(h - 1 - poc_y)}
    cols = ["open_time", "open", "high", "low", "close"] + (["volume"] if "volume" in df else [])
    return df[cols], found
