"""ИИ «значимость»: какие паттерны рынок ДЕЙСТВИТЕЛЬНО отрабатывает, а какие — шум.

Нейросеть-«глаз» (chartnet) умеет узнавать ФОРМУ: «это похоже на FVG». Но среди сотни FVG
реакцию дают единицы. Правила («импульс + слом структуры + объём») угадывают это грубо.
Поэтому учим вторую модель — на ИСХОДАХ, которые рынок уже показал на истории:

  1) МЕТКИ. Для каждого случая на истории смотрим, что было ПОТОМ — как в сделке со стопом:
       • FVG / OB / Breaker / Mitigation — цена впервые пришла в зону; стоп за дальней границей;
       • BOS / CHoCH — стоп обратно за пробитый уровень;
       • EQH / EQL — ликвидность сняли, ждём разворот; стоп 1 ATR за уровнем;
       • двойное дно / вершина, ГиП — стоп за дном (вершиной) фигуры.
     Реакция = цена прошла 2 риска (фигуры — 1.5) раньше, чем закрылась за стопом (20 свечей,
     фигуры — 40; при широком стопе — больше). Пропорция всегда одна — на случайном графике шанс у всех одинаковый,
     и модель учит рынок, а не арифметику стопа.
     Реакция = 1, нет реакции = 0. Сами ставим метки — тысячи случаев, без ручной разметки.
  2) ПРИЗНАКИ — только то, что было известно В МОМЕНТ входа (будущее не подглядываем):
     размер в ATR, импульс, объём, слом структуры, тренд, где цена в диапазоне (дорого/дёшево),
     как далеко цена уходила от зоны, сколько свечей прошло… и (если есть) 64 числа «вида»
     от chartnet.
  3) МОДЕЛЬ. Логистическая регрессия или маленькая нейросеть — выбирается по проверке.
     Учится в setup_ai.py, в боте считает на чистом numpy: веса в models/quality.json.
  4) ЭКЗАМЕН — на ДРУГИХ монетах и на последних по времени свечах. Модель паттерна
     включается, только если её «сильные» находки реально чаще дают реакцию, чем в среднем.

В боте значимость — одна из частей оценки силы находки (strength.py): сдвигает оценку
эксперта. Чисел пользователю бот не показывает.
"""

import json
import math
import os

import numpy as np

from structure import atr

PATH = os.path.join("models", "quality.json")
REWARD = 2.0               # реакция = цена прошла в нужную сторону 2 «риска» (риск — до стопа, ≥ 1 ATR)…
HORIZON = 20               # …за 20 свечей, не закрывшись за стопом
SHAPE_HORIZON = 40         # фигурам даём больше времени…
SHAPE_REWARD = 1.5         # …и цель 1.5 риска (стоп — за дном / вершиной фигуры)
MAX_HORIZON = 150          # широкий стоп — больше времени, но не больше 150 свечей
CONTEXT = 100              # «где цена в диапазоне» — за 100 свечей
EMBED_BEFORE = 40
MIN_TRAIN = 150            # меньше случаев — не учим
MIN_EXAM = 100
MIN_AUC = 0.55             # модель должна различать лучше монетки (0.5)…
MIN_LIFT = 0.05            # …и её «сильные» находки — на 5 п.п. чаще с реакцией, чем в среднем,
MIN_Z = 2.0                # …и это не везение на маленькой выборке (≈ 95% уверенности)
WEAK_SHARE = 0.35          # «обычная» строгость прячет 35% самых слабых
STRONG_SHARE = 0.70        # ★ сильные — лучшие 30%
STRICTNESS = ("normal", "strict", "off")
BLOCK_KINDS = ("ob", "breaker", "mitigation")
SHAPE_NAMES = ("Двойное дно", "Двойная вершина", "Голова и плечи", "Перевёрнутая ГиП")
BULLISH_SHAPES = ("Двойное дно", "Перевёрнутая ГиП")
BEARISH_SHAPES = ("Двойная вершина", "Голова и плечи")
FAMILY_TEXT = {"fvg": "FVG", "block": "OB / Breaker / Mitigation", "structure": "BOS / CHoCH",
               "eq": "EQH / EQL", "shape": "двойное дно / вершина, ГиП"}

MODELS = {}                # семья → модель (загружается из models/quality.json)
EMBED = {"net": None}      # chartnet для «вида», если модель его использует


# --- к какой семье относится находка --------------------------------------------------------------
def family_of(item):
    kind = str(item.get("kind", ""))
    if kind.endswith("_fvg"):
        return "fvg"
    if kind.split("_")[-1] in BLOCK_KINDS:
        return "block"
    if kind in ("bos", "choch"):
        return "structure"
    if kind in ("eqh", "eql"):
        return "eq"
    if item.get("neckline") and item.get("points"):
        return "shape"
    return None


def is_bullish(item):
    kind = str(item.get("kind", ""))
    if kind in ("eqh", "eql"):
        return kind == "eql"                       # после снятия лоёв ждём рост
    if kind.startswith(("bullish", "bearish")):
        return kind.startswith("bullish")
    if item.get("direction") in ("up", "down"):
        return item["direction"] == "up"
    if item.get("points"):                         # фигура: двойное дно / перевёрнутая ГиП — бычьи
        if item.get("name") in BULLISH_SHAPES:
            return True
        if item.get("name") in BEARISH_SHAPES:
            return False
        points = item["points"]
        if len(points) >= 6:                       # ГиП: голова ниже соседних точек — перевёрнутая
            return points[2][1] < points[1][1]
        return points[1][1] < points[0][1]
    return True


# --- свечи один раз в numpy ----------------------------------------------------------------------
class Arrays:
    def __init__(self, df):
        self.open = df["open"].to_numpy(float)
        self.high = df["high"].to_numpy(float)
        self.low = df["low"].to_numpy(float)
        self.close = df["close"].to_numpy(float)
        self.n = len(df)
        sizes = atr(df).to_numpy(float)
        fallback = np.nanmean(self.high - self.low) or 1e-9
        self.atr = np.where(np.isfinite(sizes) & (sizes > 0), sizes, fallback)
        self.ema = df["close"].ewm(span=50, adjust=False).mean().to_numpy(float)
        if "volume" in df:
            volume = df["volume"].astype(float)
            average = volume.rolling(20, min_periods=5).mean().shift(1)
            self.volume = (volume / average.where(average > 0)).to_numpy(float)
        else:
            self.volume = None
        self.df = df


_arrays_cache = {}


def arrays(df):
    key = id(df)
    cached = _arrays_cache.get(key)
    if cached is None or cached.df is not df or cached.n != len(df):
        if len(_arrays_cache) > 64:
            _arrays_cache.clear()
        cached = _arrays_cache[key] = Arrays(df)
    return cached


def _log(x):
    return math.log1p(max(0.0, float(x)))


def _vol(a, i):
    if a.volume is None or not 0 <= i < a.n or not np.isfinite(a.volume[i]):
        return 0.0
    return _log(a.volume[i])


# --- признаки: что было известно в момент t -------------------------------------------------------
def context(a, end, t, bullish):
    """Общий фон: тренд, где цена в диапазоне, волатильность, как давно и далеко."""
    s = 1.0 if bullish else -1.0
    size = a.atr[t]
    t50 = max(0, t - 50)
    lo = a.low[max(0, t - CONTEXT):t + 1].min()
    hi = a.high[max(0, t - CONTEXT):t + 1].max()
    position = (a.close[t] - lo) / (hi - lo) if hi > lo else 0.5
    calm = a.atr[max(0, t - CONTEXT):t + 1].mean()
    t5 = max(0, t - 5)
    return [
        np.clip(s * (a.close[t] - a.close[t50]) / size / 10, -3, 3),     # тренд за 50 свечей
        np.clip(s * (a.close[t] - a.ema[t]) / size / 5, -3, 3),          # над/под EMA50
        position if bullish else 1 - position,                           # 0 — «дёшево» для паттерна
        np.clip(a.atr[t] / calm if calm > 0 else 1, 0, 4),              # волатильность сейчас
        _log(t - end) / 3,                                               # сколько свечей прошло
        np.clip(s * (a.close[t5] - a.close[t]) / size / 3, -3, 3),       # как быстро шли к паттерну
    ]


CONTEXT_NAMES = ["тренд", "EMA50", "дёшево/дорого", "волатильность", "возраст", "подход"]


def _away(a, zone, end, t, bullish):
    """Как далеко цена уходила от зоны до возвращения (в ATR): свежая «нетронутая» зона сильнее."""
    bottom, top = zone
    if t <= end:
        return 0.0
    if bullish:
        move = a.high[end + 1:t + 1].max() - top
    else:
        move = bottom - a.low[end + 1:t + 1].min()
    return _log(max(0.0, move) / a.atr[t]) / 2


def family_features(family, a, item, t):
    bullish = is_bullish(item)
    start, end = int(item["start"]), int(min(item.get("end", item["start"]), a.n - 1))
    size = a.atr[min(end, a.n - 1)]
    if family == "fvg":
        bottom, top = item["zone"]
        middle = min(start + 1, a.n - 1)
        body = abs(a.close[middle] - a.open[middle])
        candle = a.high[middle] - a.low[middle]
        return [np.clip((top - bottom) / size, 0, 5) / 2, np.clip(body / size, 0, 6) / 3,
                body / candle if candle > 0 else 0, float(bool(item.get("significant"))),
                _vol(a, middle), _away(a, item["zone"], end, t, bullish)]
    if family == "block":
        bottom, top = item["zone"]
        kind = str(item.get("kind", "")).split("_")[-1]
        ratio = item.get("volume_ratio")
        return [np.clip((top - bottom) / size, 0, 5) / 2, np.clip(item.get("displacement", 0) or 0, 0, 8) / 4,
                float(bool(item.get("imbalance"))), _log(ratio) if ratio and ratio > 0 else 0.0,
                *(float(kind == k) for k in BLOCK_KINDS), _log(end - start) / 3,
                _away(a, item["zone"], end, t, bullish)]
    if family == "structure":
        level = item["level"]
        leg = a.high[start:end + 1].max() - a.low[start:end + 1].min()
        return [np.clip(abs(a.close[end] - level) / size, 0, 5), np.clip(leg / size, 0, 30) / 10,
                _log(end - start) / 3, float(item.get("kind") == "choch"), _vol(a, end)]
    if family == "eq":
        touches = [x for x in item.get("touches", []) if x < a.n]
        values = (a.low if item["kind"] == "eql" else a.high)[touches] if touches else np.array([0.0])
        return [np.clip((values.max() - values.min()) / size, 0, 3), min(len(touches), 6) / 3,
                _log(end - start) / 3]
    if family == "shape":
        x0, y0, x1, y1 = item["neckline"]
        ys = [y for _, y in item["points"][:-1]]
        height = (item.get("height") or (max(ys) - min(ys))) / size
        return [np.clip(height, 0, 30) / 10, float(bool(item.get("confirmed"))),
                float(len(item["points"]) >= 6),                       # ГиП, а не двойное
                _log(item.get("window", end - start)) / 4]
    raise ValueError(family)


def _end(item, n):
    return int(min(item.get("end", item["start"] + item.get("window", 1) - 1), n - 1))


def decision_bar(family, item, n):
    """Последняя свеча, известная в момент решения «сейчас»: для зон и ликвидности — текущая
    (цена подходит к зоне), для слома и фигуры — свеча, на которой они сложились."""
    return _end(item, n) if family in ("structure", "shape") else n - 1


def features(family, df_or_arrays, item, t=None):
    """Вектор признаков находки по свечам ДО t включительно (будущее не видно)."""
    a = df_or_arrays if isinstance(df_or_arrays, Arrays) else arrays(df_or_arrays)
    t = decision_bar(family, item, a.n) if t is None else int(t)
    end = _end(item, a.n)
    bullish = is_bullish(item)
    return np.array(context(a, end, t, bullish) + family_features(family, a, item, t), dtype=float)


def embedding(df, item, t=None, net=None):
    """64 числа «вида» от chartnet: как выглядел график перед входом (бычий вид — медвежий переворачиваем)."""
    import visual
    net = net or EMBED["net"]
    if net is None:
        return None
    n = len(df)
    t = decision_bar(family_of(item), item, n) if t is None else int(t)
    ohlc = visual.normalize_ohlc(df, max(0, int(item["start"]) - EMBED_BEFORE), t + 1,
                                 mirror=not is_bullish(item))
    if ohlc is None or len(ohlc) < 15:
        return np.zeros(64)
    return np.asarray(net(ohlc), dtype=float)


# --- метки: что сделал рынок ПОТОМ ----------------------------------------------------------------
def reaction(a, t, bullish, entry, stop, target, horizon=HORIZON):
    """Начиная со свечи t: 1 — цена дошла до entry ± target раньше, чем закрылась за stop;
    0 — нет; None — график закончился раньше, чем стало понятно."""
    for x in range(t, min(t + horizon, a.n)):
        if bullish:
            if a.close[x] < stop:
                return 0
            if a.high[x] >= entry + target:
                return 1
        else:
            if a.close[x] > stop:
                return 0
            if a.low[x] <= entry - target:
                return 1
    return 0 if t + horizon <= a.n else None


def first_touch(a, zone, end, bullish):
    bottom, top = zone
    after = a.low[end + 1:] <= top if bullish else a.high[end + 1:] >= bottom
    hits = np.flatnonzero(after)
    return int(end + 1 + hits[0]) if len(hits) else None


def label(family, a, item):
    """(t, метка 0/1): t — последняя свеча ДО входа (по ней считаем признаки), исход — с t + 1.
    None — исход ещё неизвестен.
    Риск (до стопа) и цель всегда в одной пропорции 1 : REWARD — так на «случайном» графике
    шанс одинаков для любых паттернов, и модель учит рынок, а не арифметику стопа."""
    bullish = is_bullish(item)
    end = _end(item, a.n)
    sign = 1 if bullish else -1
    horizon = HORIZON
    reward = REWARD
    if family in ("fvg", "block"):
        bottom, top = item["zone"]
        touch = item.get("touched_at") if family == "block" else first_touch(a, item["zone"], end, bullish)
        if touch is None or touch >= a.n or touch <= end:
            return None
        t = int(touch) - 1
        entry = top if bullish else bottom
        risk = max(top - bottom, a.atr[t])             # стоп — за дальней границей зоны
    elif family == "structure":
        t = end
        entry = a.close[t]
        risk = max(abs(entry - item["level"]) + 0.5 * a.atr[t], a.atr[t])   # стоп — обратно за уровень
    elif family == "eq":
        swept = item.get("swept_at")
        if swept is None or swept <= end:
            return None
        t = int(swept) - 1
        entry = float(item.get("level", a.low[swept] if bullish else a.high[swept]))
        risk = a.atr[t]
    elif family == "shape":
        t = end
        ys = [y for _, y in item["points"][:-1]]
        entry = a.close[t]
        extreme = min(ys) if bullish else max(ys)      # стоп — за дном (вершиной) фигуры
        risk = max(abs(entry - extreme), a.atr[t])
        horizon, reward = SHAPE_HORIZON, SHAPE_REWARD
    else:
        return None
    # чем дальше стоп, тем больше времени нужно цене (как у случайного блуждания: время ~ путь²)
    horizon = int(min(MAX_HORIZON, horizon * max(1.0, risk / a.atr[t]) ** 2))
    result = reaction(a, t + 1, bullish, entry, entry - sign * risk, reward * risk, horizon)
    return None if result is None else (t, result)


# --- случаи с истории -----------------------------------------------------------------------------
def candidates(df, families=None):
    """Все случаи паттернов на графике (и прошлые, и отработанные): [(семья, находка)]."""
    from blocks import all_blocks
    from smc import find_all_fvg
    from structure import equal_levels, structure_events
    families = families or list(FAMILY_TEXT)
    found = []
    if "fvg" in families:
        found += [("fvg", f) for f in find_all_fvg(df, lookback=len(df), include_filled=True)]
    if "block" in families:
        found += [("block", b) for b in all_blocks(df) if b["family"] != "ob" or b.get("strong", True)]
    if "structure" in families:
        found += [("structure", e) for e in structure_events(df)]
    if "eq" in families:
        found += [("eq", e) for e in equal_levels(df)]
    if "shape" in families:
        from history import history_shapes
        from scanner import WINDOWS, MIN_SCORE
        from templates import template_shape
        for name in SHAPE_NAMES:
            for f in history_shapes(df, template_shape(name), WINDOWS, MIN_SCORE, name):
                if f.get("neckline"):
                    found.append(("shape", {**f, "name": name}))
    return found


def cases(df, families=None, net=None):
    """Случаи с метками: [(семья, признаки, «вид» или None, метка, время входа)]."""
    a = arrays(df)
    result = []
    for family, item in candidates(df, families):
        try:
            labelled = label(family, a, item)
            if labelled is None:
                continue
            t, y = labelled
            x = features(family, a, item, t)
            e = embedding(df, item, t, net) if net is not None else None
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if np.all(np.isfinite(x)):
            result.append((family, x, e, y, t / max(1, a.n - 1)))
    return result


# --- модели на numpy -----------------------------------------------------------------------------
def _standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1
    return mean, std


def train_logistic(X, y, l2=0.05, steps=600, rate=0.3):
    mean, std = _standardize(X)
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1])
    b = math.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= rate * (Z.T @ (p - y) / len(y) + l2 * w)
        b -= rate * float(np.mean(p - y))
    return {"type": "logistic", "mean": mean.tolist(), "std": std.tolist(), "w": w.tolist(), "b": float(b)}


def train_mlp(X, y, seed=0):
    from sklearn.neural_network import MLPClassifier          # только для обучения
    mean, std = _standardize(X)
    net = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=1e-2, max_iter=400,
                        early_stopping=True, validation_fraction=0.2, random_state=seed)
    net.fit((X - mean) / std, y)
    return {"type": "mlp", "mean": mean.tolist(), "std": std.tolist(),
            "w1": net.coefs_[0].tolist(), "b1": net.intercepts_[0].tolist(),
            "w2": net.coefs_[1][:, 0].tolist(), "b2": float(net.intercepts_[1][0])}


def predict(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["std"])
    if model["type"] == "logistic":
        logit = Z @ np.asarray(model["w"]) + model["b"]
    else:
        hidden = np.maximum(Z @ np.asarray(model["w1"]) + np.asarray(model["b1"]), 0)
        logit = hidden @ np.asarray(model["w2"]) + model["b2"]
    return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))


def auc(y, p):
    """Насколько модель отличает «с реакцией» от «без»: 0.5 — монетка, 1 — идеально."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    _, groups, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.bincount(groups, ranks)[groups] / counts[groups]          # одинаковым — средний ранг
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(y, p):
    """Экзамен: база (как часто реакция вообще), у сильных (лучшие 30%) и у слабых (худшие 35%)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p)
    order = np.argsort(p)
    weak = order[:max(1, int(len(p) * WEAK_SHARE))]
    strong = order[int(len(p) * STRONG_SHARE):]
    base = float(y.mean())
    strong_rate = float(y[strong].mean()) if len(strong) else 0.0
    spread = math.sqrt(max(base * (1 - base), 1e-9) / max(1, len(strong)))
    return {"cases": int(len(y)), "base": base, "auc": auc(y, p), "strong": strong_rate,
            "weak": float(y[weak].mean()) if len(weak) else 0.0,
            "z": (strong_rate - base) / spread}           # не случайность ли: z ≥ 2 — почти точно нет


def _matrix(rows, embed):
    return np.array([np.concatenate([x, e]) if embed and e is not None else x for _, x, e, _, _ in rows])


def fit_family(train_rows, exam_rows, use_mlp=True):
    """Учит модель одной семьи: варианты (логистическая / нейросеть, с «видом» chartnet или без),
    выбор — на последних по времени 20% обучающих случаев, экзамен — на других монетах."""
    train_rows = sorted(train_rows, key=lambda r: r[4])
    y = np.array([r[3] for r in train_rows], dtype=float)
    if len(train_rows) < MIN_TRAIN or y.min() == y.max():
        return None
    cut = int(len(train_rows) * 0.8)
    has_embed = all(r[2] is not None for r in train_rows + exam_rows)
    best = None
    for embed in ([False, True] if has_embed else [False]):
        X = _matrix(train_rows, embed)
        trainers = [train_logistic] + ([train_mlp] if use_mlp else [])
        for trainer in trainers:
            try:
                inner = trainer(X[:cut], y[:cut])
            except Exception:                                  # нет sklearn и т.п.
                continue
            score = auc(y[cut:], predict(inner, X[cut:]))
            if best is None or score > best[0] + 0.005:
                best = (score, trainer, embed)
    if best is None:
        return None
    _, trainer, embed = best
    X = _matrix(train_rows, embed)
    model = trainer(X, y)
    model["embed"] = bool(embed)
    train_p = predict(model, X)
    model["weak"] = float(np.quantile(train_p, WEAK_SHARE))
    model["strong"] = float(np.quantile(train_p, STRONG_SHARE))
    model["train"] = report(y, train_p)
    if exam_rows:
        ye = np.array([r[3] for r in exam_rows], dtype=float)
        model["exam"] = report(ye, predict(model, _matrix(exam_rows, embed)))
    model["on"] = decide(model.get("exam"))
    return model


def decide(exam):
    """Включаем, только если на чужих монетах модель реально отделяет значимые от шума."""
    return bool(exam and exam["cases"] >= MIN_EXAM and exam["auc"] >= MIN_AUC
                and exam["strong"] >= exam["base"] + MIN_LIFT and exam.get("z", 0) >= MIN_Z)


def split_by_time(rows, share=0.7):
    """Экзамен честный и по времени: учимся на первых 70% истории, экзаменуемся на последних 30%."""
    return [r for r in rows if r[4] < share], [r for r in rows if r[4] >= share]


def train_all(train_frames, exam_frames, net=None, families=None, use_mlp=True, say=print):
    """train_frames / exam_frames — графики (DataFrame) РАЗНЫХ монет. Возвращает {семья: модель}."""
    families = families or list(FAMILY_TEXT)
    train_rows = [r for df in train_frames for r in cases(df, families, net)]
    exam_rows = [r for df in exam_frames for r in cases(df, families, net)]
    train_rows, _ = split_by_time(train_rows)
    _, exam_rows = split_by_time(exam_rows, 0.5)       # экзамен — вторая половина времени
    models = {}
    for family in families:
        tr = [r for r in train_rows if r[0] == family]
        ex = [r for r in exam_rows if r[0] == family]
        model = fit_family(tr, ex, use_mlp)
        if model is None:
            say(f"   {FAMILY_TEXT[family]}: мало случаев ({len(tr)}) — пропускаю")
            continue
        models[family] = model
        say("   " + exam_text(family, model))
    return models


def exam_text(family, model):
    exam = model.get("exam") or {}
    if not exam:
        return f"{FAMILY_TEXT[family]}: нет экзамена"
    mark = "✅ включаю" if model.get("on") else "— пока недостаточно уверенно, не включаю"
    return (f"{FAMILY_TEXT[family]}: реакция в среднем {exam['base']:.0%}, у сильных по ИИ "
            f"{exam['strong']:.0%}, у слабых {exam['weak']:.0%} ({exam['cases']} случаев, "
            f"AUC {exam['auc']:.2f}) {mark}")


# --- в боте ---------------------------------------------------------------------------------------
def save(models, path=PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False)


def load(path=PATH, chartnet_path=os.path.join("models", "chartnet.npz")):
    """При запуске бота: модели значимости (если setup_ai.py их обучил). Возвращает включённые семьи."""
    MODELS.clear()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            models = json.load(f)
        if any(m.get("embed") for m in models.values()):
            import chartnet
            EMBED["net"] = chartnet.load(chartnet_path)
        MODELS.update(models)
    except Exception as error:
        print("Модель значимости не загрузилась, работаю по правилам:", error)
        MODELS.clear()
    return [f for f, m in MODELS.items() if m.get("on")]


def use(models, net=None):
    """Поставить модели напрямую (тесты)."""
    MODELS.clear()
    MODELS.update(models)
    EMBED["net"] = net


def strictness(settings):
    value = (settings or {}).get("quality", "normal")
    return value if value in STRICTNESS else "normal"


def enabled(family, settings=None):
    model = MODELS.get(family)
    return bool(model and model.get("on") and strictness(settings) != "off")


def score(df, item, t=None):
    """Шанс реакции 0..1 по ИИ (или None — модели нет). Пишет в находку quality и quality_level."""
    family = family_of(item)
    model = MODELS.get(family)
    if model is None or not model.get("on"):
        return None
    if "quality" in item and t is None:              # уже посчитано для этой находки
        return item["quality"]
    try:
        x = features(family, df, item, t)
        if model.get("embed"):
            e = embedding(df, item, t)
            if e is None:
                return None
            x = np.concatenate([x, e])
        p = float(predict(model, x)[0])
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    item["quality"] = p
    item["quality_base"] = (model.get("exam") or model.get("train") or {}).get("base")
    item["quality_level"] = "strong" if p >= model["strong"] else "weak" if p < model["weak"] else "normal"
    return p
