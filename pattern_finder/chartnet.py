"""Своя нейросеть, обученная на ГРАФИКАХ (а не на фотографиях кошек, как DINOv2).

Зачем: экзамен показал, что готовая нейросеть хуже простого HOG — она не знает,
как выглядят Order Block, FVG или Range. Значит, учим свою:

  1) ДАННЫЕ. Бот находит на истории Binance тысячи настоящих паттернов своими правилами
     (Order Block, FVG, BOS, Range, флаги, каналы, треугольники…). Каждый случай —
     участок свечей (40 до, паттерн, 20 после), цена и время растянуты в 0..1.
     Медвежьи случаи переворачиваем в бычьи — учим форму, а не направление.
  2) ВХОД. Участок → 64 точки open, 64 high, 64 low, 64 close = 256 чисел.
     Чуть-чуть «шевелим» данные (обрезаем края на пару свечей, добавляем шум) —
     так сеть учится не запоминать конкретные случаи, а узнавать форму.
  3) СЕТЬ. Многослойный перцептрон 256 → 128 → 64 → «какой это паттерн».
     Учим угадывать название паттерна. Сам ответ нам не нужен: нужен предпоследний
     слой — 64 числа. Чтобы угадывать, сеть вынуждена складывать туда суть формы,
     поэтому у похожих паттернов эти 64 числа похожи. Это и есть «описание вида» (эмбеддинг).
  4) ЭКЗАМЕН — на ДРУГИХ монетах, которых сеть не видела (eye_check.py), против HOG.

Учится один раз (scikit-learn, только для обучения), а в боте работает на чистом numpy:
веса лежат в models/chartnet.npz.
"""

import numpy as np

POINTS = 64                 # точек на каждую из open/high/low/close
HIDDEN = (128, 64)
AUGMENT = 3                 # сколько «пошевеленных» копий каждого случая
TRAIN_PATTERNS = ["Order Block", "Breaker Block", "Mitigation Block", "FVG", "BOS", "CHoCH",
                  "EQH / EQL", "Range", "Флаг", "Вымпел", "Треугольник", "Клин", "Канал",
                  "Двойное дно", "Двойная вершина", "Голова и плечи"]


def to_vector(ohlc, points=POINTS):
    """Участок свечей (n×4, 0..1) → 256 чисел: каждый столбец растянут до 64 точек."""
    ohlc = np.asarray(ohlc, dtype=float)
    x_old = np.linspace(0, 1, len(ohlc))
    x_new = np.linspace(0, 1, points)
    return np.concatenate([np.interp(x_new, x_old, ohlc[:, k]) for k in range(4)]).astype(np.float32)


def _renormalize(ohlc):
    low, high = ohlc[:, 2].min(), ohlc[:, 1].max()
    return (ohlc - low) / (high - low) if high > low else None


def augment(ohlc, rng, copies=AUGMENT):
    """«Пошевеленные» копии: обрезаем по краям до 5 свечей и добавляем лёгкий шум."""
    result = []
    n = len(ohlc)
    for _ in range(copies):
        left, right = rng.integers(0, 6), rng.integers(0, 6)
        part = ohlc[left:n - right] if n - left - right >= 15 else ohlc
        part = part + rng.normal(0, 0.01, part.shape)
        part = _renormalize(part)
        if part is not None:
            result.append(part)
    return result


def dataset(windows, seed=0, copies=AUGMENT):
    """[(паттерн, свечи)] → X (вектора), y (номера паттернов), список названий."""
    rng = np.random.default_rng(seed)
    names = sorted({name for name, _ in windows})
    X, y = [], []
    for name, ohlc in windows:
        for part in [ohlc] + augment(ohlc, rng, copies):
            X.append(to_vector(part))
            y.append(names.index(name))
    return np.array(X), np.array(y), names


def train(windows, hidden=HIDDEN, seed=0, max_iter=300):
    """Обучает сеть. Возвращает веса для ChartNet (только скрытые слои — они и дают «описание»)."""
    from sklearn.neural_network import MLPClassifier         # нужен только для обучения
    X, y, names = dataset(windows, seed)
    net = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3,
                        early_stopping=True, validation_fraction=0.15, max_iter=max_iter,
                        random_state=seed)
    net.fit(X, y)
    return {"weights": [w.astype(np.float32) for w in net.coefs_[:len(hidden)]],
            "biases": [b.astype(np.float32) for b in net.intercepts_[:len(hidden)]],
            "names": names, "cases": len(windows), "accuracy": float(net.best_validation_score_ or 0)}


class ChartNet:
    """Обученная сеть в боте: свечи → 64 числа (выход последнего скрытого слоя), длина 1."""

    def __init__(self, model):
        self.weights = model["weights"]
        self.biases = model["biases"]
        self.cases = int(model.get("cases", 0))

    def __call__(self, ohlc):
        h = to_vector(ohlc)
        for w, b in zip(self.weights, self.biases):
            h = np.maximum(h @ w + b, 0)                        # ReLU
        norm = np.linalg.norm(h)
        return h / norm if norm else h


def save(model, path):
    arrays = {f"w{i}": w for i, w in enumerate(model["weights"])}
    arrays.update({f"b{i}": b for i, b in enumerate(model["biases"])})
    np.savez(path, names=np.array(model["names"]), cases=model["cases"], **arrays)


def load(path):
    data = np.load(path)
    count = len([k for k in data.files if k.startswith("w")])
    return ChartNet({"weights": [data[f"w{i}"] for i in range(count)],
                     "biases": [data[f"b{i}"] for i in range(count)], "cases": int(data["cases"])})
