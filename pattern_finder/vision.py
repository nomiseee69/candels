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
    return img


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


def find_lines(fg):
    """Ищет длинные прямые (нарисованные линии), склеивает дубли, убирает пунктир."""
    h, w = fg.shape
    raw = cv2.HoughLinesP(
        fg, 1, np.pi / 180, threshold=60,
        minLineLength=int(w * MIN_LINE_RATIO), maxLineGap=MAX_LINE_GAP,
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
        if _fill_ratio(fg, ln) >= DOTTED_FILL and _side_fill(fg, ln) <= SIDE_FILL_MAX
    ]
    return sorted(lines, key=lambda ln: ln.x1)


def erase_lines(mask, lines, thickness=11):
    clean = mask.copy()
    for ln in lines:
        cv2.line(clean, (ln.x1, ln.y1), (ln.x2, ln.y2), 0, thickness)
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
