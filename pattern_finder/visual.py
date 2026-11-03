"""«Глаз» бота: насколько два участка графика ПОХОЖИ НА ВИД.

Твой скриншот и график с биржи бот рисует одинаково — в маленькую чёрно-белую картинку
96×48 (свечи: тень — тонкая, тело — толще; цена и время растянуты на весь кадр).
Цвета темы, масштаб и размер скрина при этом пропадают — остаётся только форма.

Потом картинка превращается в набор чисел — «описание вида». Сейчас это HOG
(гистограммы направлений краёв: где на картинке наклонные, вертикальные, горизонтальные
края) — классический приём компьютерного зрения. Похожие картинки → похожие числа;
сравниваем косинусом угла между наборами.

Следующий шаг — вместо HOG поставить нейросеть (функция describe): она тоже превращает
картинку в числа, только «понимает» вид лучше. Остальной код не изменится.
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 96, 48
HOG = cv2.HOGDescriptor((WIDTH, HEIGHT), (16, 16), (8, 8), (8, 8), 9)
RANDOM_COS = 0.45     # косинус для двух случайных участков рынка (медиана ~0.4) — это «0% похожести»


def normalize_ohlc(df, start=0, end=None, mirror=False):
    """Свечи участка → массив (n, 4) open/high/low/close в долях 0..1 от размаха цены.
    mirror — перевернуть ценой вниз (бычий пример сравнить с медвежьей находкой)."""
    part = df.iloc[start:end]
    ohlc = part[["open", "high", "low", "close"]].to_numpy(dtype=float)
    low, high = ohlc[:, 2].min(), ohlc[:, 1].max()
    if high - low <= 0:
        return None
    ohlc = (ohlc - low) / (high - low)
    if mirror:
        ohlc = 1 - ohlc[:, [0, 2, 1, 3]]          # high и low меняются местами
    return ohlc


def render(ohlc, width=WIDTH, height=HEIGHT):
    """Свечи → чёрно-белая картинка height×width (как маленький скриншот)."""
    img = np.zeros((height, width), np.uint8)
    n = len(ohlc)
    step = width / n
    body = max(1, int(step * 0.6))
    y = lambda v: int(round((1 - v) * (height - 1)))
    for i, (o, h, lo, c) in enumerate(ohlc):
        x = int((i + 0.5) * step)
        cv2.line(img, (x, y(h)), (x, y(lo)), 140, 1)
        top, bottom = y(max(o, c)), y(min(o, c))
        cv2.rectangle(img, (x - body // 2, top), (x - body // 2 + body - 1, bottom), 255, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def describe(img):
    """Картинка → «описание вида» (вектор чисел длины 1). Место для нейросети."""
    vector = HOG.compute(img).ravel().astype(float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def look(ohlc):
    return describe(render(ohlc)) if ohlc is not None and len(ohlc) >= 5 else None


def similarity(a, b):
    """Похожесть двух описаний 0..1 (косинус, пересчитанный от «как у случайных»)."""
    if a is None or b is None:
        return None
    cos = float(np.dot(a, b))
    return float(np.clip((cos - RANDOM_COS) / (1 - RANDOM_COS), 0, 1))
