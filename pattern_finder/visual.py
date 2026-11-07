"""«Глаз» бота: насколько два участка графика ПОХОЖИ НА ВИД.

Твой скриншот и график с биржи бот рисует одинаково — сам, из свечей. Цвета темы,
масштаб и размер скрина пропадают — остаётся только форма. Потом картинка
превращается в набор чисел — «описание вида». Похожие картинки → похожие числа;
сравниваем косинусом угла между наборами.

Два «глаза»:
  • HOG — классическое компьютерное зрение: гистограммы направлений краёв
    (картинка 96×48, чёрно-белая). Работает всегда, ничего не нужно ставить.
  • Нейросеть — готовая обученная модель DINOv2 (или MobileNet) в формате ONNX.
    Картинка 224×224 «как в TradingView» (тёмный фон, зелёные и красные свечи).
    Сеть видела миллионы картинок и описывает форму целиком, а не только края.
    Ставится командой  python setup_ai.py  — она скачает модель, САМА проверит
    на реальном рынке, видит ли нейросеть похожие паттерны лучше HOG, и включит её,
    только если лучше (решение — в ai.json).

«Похожесть» = (косинус − как у двух случайных участков) / (1 − как у случайных):
у каждого глаза своя «случайная» планка, её меряет setup_ai.py.
"""

import json
import os
from collections import OrderedDict

import cv2
import numpy as np

WIDTH, HEIGHT = 96, 48
CELL, BINS = 8, 9          # HOG: клетки 8×8 пикселей, 9 направлений краёв (0–180°)
HOG_RANDOM_COS = 0.40      # косинус для двух случайных участков рынка у HOG (медиана ~0.35)
NEURAL_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # как при обучении сети (ImageNet)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIG = "ai.json"
CACHE_SIZE = 5000          # одинаковые участки не прогоняем через глаз дважды


# --- свечи → картинка --------------------------------------------------------------------------
def normalize_ohlc(df, start=0, end=None, mirror=False):
    """Свечи участка → массив (n, 4) open/high/low/close в долях 0..1 от размаха цены.
    mirror — перевернуть ценой вниз (бычий пример сравнить с медвежьей находкой)."""
    part = df.iloc[start:end]
    ohlc = part[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if len(ohlc) == 0:
        return None
    low, high = ohlc[:, 2].min(), ohlc[:, 1].max()
    if high - low <= 0:
        return None
    ohlc = (ohlc - low) / (high - low)
    if mirror:
        ohlc = 1 - ohlc[:, [0, 2, 1, 3]]          # high и low меняются местами
    return ohlc


def render(ohlc, width=WIDTH, height=HEIGHT):
    """Свечи → чёрно-белая картинка height×width (для HOG)."""
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


def render_color(ohlc, size=NEURAL_SIZE):
    """Свечи → цветная картинка size×size «как в TradingView» (для нейросети), RGB."""
    img = np.full((size, size, 3), (19, 23, 34), np.uint8)
    n = len(ohlc)
    pad = size * 0.05
    step = (size - 2 * pad) / n
    body = max(1, int(step * 0.7))
    y = lambda v: int(round(pad + (1 - v) * (size - 1 - 2 * pad)))
    for i, (o, h, lo, c) in enumerate(ohlc):
        color = (8, 153, 129) if c >= o else (242, 54, 69)
        x = int(pad + (i + 0.5) * step)
        cv2.line(img, (x, y(h)), (x, y(lo)), color, 1)
        cv2.rectangle(img, (x - body // 2, y(max(o, c))), (x - body // 2 + body - 1, y(min(o, c))), color, -1)
    return img


def hog(img, cell=CELL, bins=BINS):
    """HOG своими руками (numpy) — не зависит от версии OpenCV.
    1) градиенты: насколько резко меняется яркость по x и по y;
    2) в каждой клетке 8×8 — гистограмма направлений краёв (9 корзин по 20°), вес — сила края;
    3) соседние 2×2 клетки нормируем вместе (L2-Hys), чтобы яркость и контраст не мешали."""
    img = img.astype(np.float32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180                      # направление без знака
    rows, cols = img.shape[0] // cell, img.shape[1] // cell
    position = angle / (180 / bins) - 0.5                              # делим вес между двумя корзинами
    low = np.floor(position).astype(int)
    share = position - low
    hist = np.zeros((rows, cols, bins), np.float32)
    cell_y = (np.arange(rows * cell) // cell)[:, None].repeat(cols * cell, 1)
    cell_x = (np.arange(cols * cell) // cell)[None, :].repeat(rows * cell, 0)
    m = magnitude[:rows * cell, :cols * cell]
    for b, weight in ((low % bins, 1 - share), ((low + 1) % bins, share)):
        np.add.at(hist, (cell_y, cell_x, b[:rows * cell, :cols * cell]), m * weight[:rows * cell, :cols * cell])
    blocks = []
    for y in range(rows - 1):
        for x in range(cols - 1):
            block = hist[y:y + 2, x:x + 2].ravel()
            block = block / (np.linalg.norm(block) + 1e-6)
            block = np.minimum(block, 0.2)                              # L2-Hys: срезаем слишком сильные края
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks)


def _unit(vector):
    vector = np.asarray(vector, dtype=float).ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


# --- глаза ------------------------------------------------------------------------------------------
def hog_describe(ohlc):
    return _unit(hog(render(ohlc)))


class NeuralEye:
    """Нейросеть в формате ONNX. session — onnxruntime.InferenceSession (или подделка в тестах)."""

    def __init__(self, session, name="нейросеть"):
        self.session = session
        self.name = name
        self.input = session.get_inputs()[0].name

    def __call__(self, ohlc):
        img = render_color(ohlc).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        batch = img.transpose(2, 0, 1)[None].astype(np.float32)          # 1×3×224×224
        out = np.asarray(self.session.run(None, {self.input: batch})[0])
        if out.ndim == 3:            # трансформер (DINOv2): [1, токены, признаки] — берём CLS-токен
            out = out[0, 0]
        elif out.ndim == 4:          # свёрточная сеть: [1, каналы, h, w] — среднее по картинке
            out = out[0].mean(axis=(1, 2))
        else:
            out = out[0]
        return _unit(out)


EYE = {"name": "HOG", "describe": hog_describe, "random_cos": HOG_RANDOM_COS}
_cache = OrderedDict()


def use_eye(name, describe, random_cos):
    """Поставить другой глаз (setup_ai.py, тесты). Кэш описаний сбрасывается."""
    EYE.update(name=name, describe=describe, random_cos=float(random_cos))
    _cache.clear()


def use_hog():
    use_eye("HOG", hog_describe, HOG_RANDOM_COS)


def load_neural(model_path):
    """ONNX-модель → глаз. Бросает исключение, если onnxruntime нет или модель не открывается."""
    import onnxruntime                                   # ставится setup_ai.py, в остальном не нужен
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return NeuralEye(session, os.path.basename(model_path))


def load_config(path=CONFIG):
    """При запуске: если setup_ai.py решил, что нейросеть лучше, — включаем её. Любая ошибка — HOG."""
    if not os.path.exists(path):
        return EYE["name"]
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        if config.get("eye") == "neural":
            eye = load_neural(config["model"].replace("\\", "/"))
            use_eye(f"нейросеть ({config.get('model_name', eye.name)})", eye, config["random_cos"])
        elif config.get("eye") == "chartnet":                    # своя сеть, обученная на графиках
            import chartnet
            eye = chartnet.load(config["model"].replace("\\", "/"))
            use_eye(f"своя нейросеть (обучена на {eye.cases} случаях)", eye, config["random_cos"])
    except Exception as error:                            # нет onnxruntime, нет файла модели…
        print("Нейросеть не загрузилась, работаю с HOG:", error)
        use_hog()
    return EYE["name"]


# --- сравнение ------------------------------------------------------------------------------------
def look(ohlc):
    """«Описание вида» участка (с кэшем) или None, если свечей слишком мало."""
    if ohlc is None or len(ohlc) < 5:
        return None
    ohlc = np.asarray(ohlc, dtype=float)
    key = (EYE["name"], np.round(ohlc, 3).tobytes())
    cached = _cache.get(key)
    if cached is None:
        cached = EYE["describe"](ohlc)
        _cache[key] = cached
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return cached


def cosine(a, b):
    return float(np.dot(a, b))


def similarity(a, b):
    """Похожесть двух описаний 0..1 (косинус, пересчитанный от «как у случайных»)."""
    if a is None or b is None:
        return None
    base = EYE["random_cos"]
    return float(np.clip((cosine(a, b) - base) / (1 - base), 0, 1))


# старое имя: describe(картинка) → HOG-описание (для совместимости)
def describe(img):
    return _unit(hog(img))
