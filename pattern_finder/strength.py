"""«Эксперт»: насколько сильна КАЖДАЯ находка — так, как её оценил бы опытный трейдер.

Работает сразу, без обучения: в него заложены критерии ТА и Smart Money. Каждый критерий —
число 0..1 («насколько выполнено»), у каждого свой вес. Сила = взвешенная сумма.

  Зоны (Order Block, Breaker, Mitigation, FVG):
    импульс       — насколько резко цена ушла от зоны (displacement);
    имбаланс      — импульс оставил FVG (у OB) / дыра крупная (у FVG);
    слом          — импульс сломал структуру (BOS);
    ликвидность   — перед зоной цена сняла прошлый свинг (собрала стопы) — признак «умных денег»;
    объём         — объём импульса выше обычного;
    дисконт       — бычья зона в нижней половине диапазона (дёшево), медвежья — в верхней;
    тренд         — зона по направлению текущей структуры;
    свежесть      — в зону ещё не возвращались (FVG не заполнен);
    размер        — зона не крошечная и не огромная (0.3–2.5 средней свечи).
  BOS / CHoCH: закрытие телом за уровнем, сила свечи слома, размах ноги, объём, снятие
    ликвидности перед сломом, совпадение с трендом.
  EQH / EQL: число касаний, ровность уровня, заметность (длина), не снята, близко к цене.
  Фигуры (двойное дно/вершина, ГиП): пробой шеи, высота, симметрия, сила движения до фигуры.
  Треугольники, клинья, каналы, флаги, боковик: касания, точность формы, флагшток, глубина отката…

Дальше поверх эксперта:
  • если setup_ai.py обучил «значимость» на истории рынка (quality.py) — она сдвигает оценку;
  • если ты ставишь 👍/👎 — личная модель (feedback.py) переучивает ВЕСА критериев и строгость
    под тебя: что для тебя важнее, а что нет.

Пользователю бот числа не показывает: сила решает, КАКИЕ находки прислать и в каком порядке.
"""

import math

import numpy as np

import quality
from structure import structure_events
from trendlines import pivot_highs, pivot_lows

SCALE = 8.0                # сила 0..1 → «логит»: 0.5 — середина, ±0.5 → ±4
STRONG = 0.66              # сильная находка — от 0.66
WEAK = 0.50                # слабая — ниже 0.50 (её прячет «обычная» строгость)
LIQUIDITY_LOOKBACK = 40
PIVOT = 3
QUALITY_WEIGHT = 6.0       # насколько «значимость по истории» сдвигает оценку эксперта

FACTOR_NAMES = {
    "impulse": "импульс", "imbalance": "имбаланс (FVG)", "structure": "слом структуры",
    "liquidity": "снятие ликвидности", "volume": "объём", "discount": "дисконт / премиум",
    "trend": "по тренду", "fresh": "свежесть зоны", "size": "размер зоны",
    "close_beyond": "закрытие за уровнем", "body": "сильная свеча слома", "leg": "размах движения",
    "touches": "касания", "tight": "ровный уровень", "obvious": "заметность уровня",
    "untapped": "ликвидность не снята", "near": "близко к цене",
    "confirmed": "пробой шеи", "height": "высота фигуры", "symmetry": "симметрия",
    "arrival": "движение до фигуры", "match": "точность формы", "pole": "флагшток",
    "shallow": "неглубокий откат", "length": "длительность", "poc": "профиль объёма (POC)",
}

PRIOR = {   # веса критериев «по учебнику» — с них начинается каждый пользователь
    "zone": {"impulse": 1.5, "imbalance": 1.2, "structure": 1.2, "liquidity": 1.2, "volume": 1.0,
             "discount": 0.8, "trend": 0.8, "fresh": 0.6, "size": 0.5},
    "structure": {"close_beyond": 1.2, "body": 1.0, "leg": 1.0, "volume": 1.0, "liquidity": 1.0,
                  "trend": 0.8},
    "eq": {"touches": 1.2, "tight": 1.2, "obvious": 0.8, "untapped": 1.5, "near": 0.8},
    "shape": {"confirmed": 1.2, "height": 1.0, "symmetry": 1.0, "arrival": 1.0, "match": 0.8},
    "figure": {"touches": 1.5, "match": 1.0, "fresh": 0.8, "height": 0.6},
    "flag": {"pole": 1.5, "shallow": 1.0, "touches": 0.8, "match": 0.8, "volume": 0.6},
    "range": {"touches": 1.5, "length": 0.8, "poc": 0.6, "height": 0.6},
}


def _clip(x):
    x = float(x)
    return 0.0 if not math.isfinite(x) else min(1.0, max(0.0, x))


def group_of(item):
    """Группа критериев находки."""
    family = quality.family_of(item)
    if family in ("fvg", "block"):
        return "zone"
    if family in ("structure", "eq", "shape"):
        return family
    kind = str(item.get("kind", ""))
    if kind == "range" or "box" in item:
        return "range"
    if "pole" in item:
        return "flag"
    if "upper" in item:
        return "figure"
    if item.get("points"):
        return "shape"
    return None


# --- общие помощники ----------------------------------------------------------------------------
class Context:
    """Всё, что нужно эксперту про график, — считается один раз на график."""

    def __init__(self, df):
        self.a = quality.arrays(df)
        self.events = structure_events(df)
        highs, lows = self.a.high.tolist(), self.a.low.tolist()
        self.peaks = np.array(pivot_highs(highs, PIVOT), dtype=int)
        self.troughs = np.array(pivot_lows(lows, PIVOT), dtype=int)

    def trend_at(self, t):
        """Направление последнего слома структуры до свечи t: "up", "down" или None."""
        last = None
        for e in self.events:
            if e["end"] > t:
                break
            last = e["direction"]
        return last

    def swept(self, start, end, bullish):
        """Перед зоной / сломом цена сняла ближайший прошлый свинг (собрала ликвидность)?"""
        a = self.a
        lo, hi = max(0, start - PIVOT), min(a.n, end + 1)
        if hi <= lo:
            return 0.0
        pivots = self.troughs if bullish else self.peaks
        prior = pivots[(pivots < lo) & (pivots >= lo - LIQUIDITY_LOOKBACK)]
        if not len(prior):
            return 0.0
        last = prior[-1]
        if bullish:
            return 1.0 if a.low[lo:hi].min() < a.low[last] else 0.0
        return 1.0 if a.high[lo:hi].max() > a.high[last] else 0.0

    def location(self, price, t, bullish):
        """1 — «выгодная» сторона диапазона за 100 свечей (бычье — внизу, медвежье — вверху)."""
        a = self.a
        lo = a.low[max(0, t - 100):t + 1].min()
        hi = a.high[max(0, t - 100):t + 1].max()
        position = (price - lo) / (hi - lo) if hi > lo else 0.5
        return _clip(1 - position if bullish else position)

    def volume(self, i, reach=0):
        v = self.a.volume
        if v is None:
            return 0.5                                   # объёма нет — не знаем
        part = v[max(0, i):min(self.a.n, i + reach + 1)]
        part = part[np.isfinite(part)]
        return _clip((part.max() - 1) / 1.5) if len(part) else 0.5


_contexts = {}


def context(df):
    key = id(df)
    cached = _contexts.get(key)
    if cached is None or cached.a.df is not df or cached.a.n != len(df):
        if len(_contexts) > 32:
            _contexts.clear()
        cached = _contexts[key] = Context(df)
    return cached


# --- критерии по группам ------------------------------------------------------------------------
def _zone(c, item):
    a = c.a
    bullish = quality.is_bullish(item)
    start = int(item["start"])
    end = int(min(item.get("end", start), a.n - 1))
    t = a.n - 1
    bottom, top = item["zone"]
    size = a.atr[min(end, a.n - 1)]
    fvg = str(item.get("kind", "")).endswith("_fvg")
    middle = min(start + 1, a.n - 1)
    if fvg:
        body = abs(a.close[middle] - a.open[middle]) / size
        impulse, imbalance = _clip(body / 2.5), _clip((top - bottom) / size / 1.2)
        structure = 1.0 if item.get("broke_structure") or item.get("significant") else 0.3
        fresh = _clip(1 - float(item.get("filled") or 0))
        volume = c.volume(middle)
    else:
        # импульс — СРАЗУ после блока: за 2 свечи ушли далеко, и тела у этих свечей крупные
        nxt = slice(start + 1, min(a.n, start + 3))
        if nxt.start < a.n:
            away = (a.high[nxt].max() - top) if bullish else (bottom - a.low[nxt].min())
            body = np.abs(a.close[nxt] - a.open[nxt]).max()
            quick = min(away / size / 2.5, body / size / 1.5)
        else:
            quick = 0.0
        impulse = _clip(0.5 * _clip((item.get("displacement") or 0) / 3) + 0.5 * quick)
        imbalance = 1.0 if item.get("imbalance") else 0.0
        family = item.get("family") or str(item.get("kind", "")).split("_")[-1]
        structure = {"ob": 1.0, "breaker": 1.0}.get(family, 0.5)
        fresh = 1.0 if item.get("touched_at") is None else 0.6
        ratio = item.get("volume_ratio")
        volume = _clip((ratio - 1) / 1.5) if ratio else c.volume(start, 2)
    height = (top - bottom) / size
    size_ok = 1.0 if 0.3 <= height <= 2.5 else _clip(height / 0.3) if height < 0.3 else _clip(2.5 / height)
    trend = c.trend_at(t)
    return {
        "impulse": impulse, "imbalance": imbalance, "structure": structure,
        "liquidity": c.swept(start, max(start, end - 1) if not fvg else start + 1, bullish),
        "volume": volume, "discount": c.location((bottom + top) / 2, t, bullish),
        "trend": 0.5 if trend is None else float(trend == ("up" if bullish else "down")),
        "fresh": fresh, "size": size_ok,
    }


def _structure(c, item):
    a = c.a
    bullish = quality.is_bullish(item)
    start, end = int(item["start"]), int(min(item["end"], a.n - 1))
    size = a.atr[end]
    body = abs(a.close[end] - a.open[end])
    candle = a.high[end] - a.low[end]
    leg = a.high[start:end + 1].max() - a.low[start:end + 1].min()
    before = c.trend_at(start)
    trend = 0.5 if before is None or item.get("kind") == "choch" else float(before == ("up" if bullish else "down"))
    return {
        "close_beyond": _clip(abs(a.close[end] - item["level"]) / size),
        "body": _clip(body / candle) if candle > 0 else 0.0,
        "leg": _clip(leg / size / 6),
        "volume": c.volume(end),
        "liquidity": c.swept(max(0, start - 10), start, not bullish),   # перед сломом сняли противоположную сторону
        "trend": trend,
    }


def _eq(c, item):
    a = c.a
    t = a.n - 1
    touches = [x for x in item.get("touches", []) if x < a.n]
    values = (a.low if item.get("kind") == "eql" else a.high)[touches] if touches else np.array([0.0])
    size = a.atr[t]
    level = float(item.get("level", values.mean()))
    return {
        "touches": _clip((len(touches) - 1) / 3),
        "tight": _clip(1 - (values.max() - values.min()) / (0.25 * size)) if len(values) > 1 else 0.5,
        "obvious": _clip((int(item.get("end", 0)) - int(item.get("start", 0))) / 40),
        "untapped": 1.0 if item.get("swept_at") is None else 0.0,
        "near": _clip(1 - abs(a.close[t] - level) / (6 * size)),
    }


def _shape(c, item):
    a = c.a
    points = item["points"]
    ys = [y for _, y in points[:-1]]
    start = int(item["start"])
    end = int(min(start + item["window"] - 1, a.n - 1))
    size = a.atr[end]
    height = item.get("height") or (max(ys) - min(ys))
    if len(points) >= 6:                                   # ГиП: плечи на одном уровне
        diff = abs(points[0][1] - points[4][1])
    else:                                                  # двойное: дно (вершины) на одном уровне
        diff = abs(points[1][1] - points[3][1])
    arrival = abs(points[0][1] - points[1][1])
    return {
        "confirmed": 1.0 if item.get("confirmed") else 0.3,
        "height": _clip(height / size / 8),
        "symmetry": _clip(1 - diff / (0.3 * height)) if height > 0 else 0.0,
        "arrival": _clip(arrival / height / 1.5) if height > 0 else 0.0,
        "match": _clip((item.get("score", 0.85) - 0.8) / 0.2),
    }


def _figure(c, item):
    a = c.a
    end = int(item["upper"][2])
    size = a.atr[min(end, a.n - 1)]
    height = abs(item["upper"][1] - item["lower"][1])
    return {
        "touches": _clip((item.get("touches", 4) - 4) / 5),
        "match": _clip((item.get("score", 0.6) - 0.7) / 0.3),
        "fresh": _clip(1 - (a.n - 1 - end) / 30),
        "height": _clip(height / size / 6),
    }


def _flag(c, item):
    a = c.a
    pole = item["pole"]
    size = a.atr[min(int(pole[2]), a.n - 1)]
    return {
        "pole": _clip((abs(pole[3] - pole[1]) / size - 4) / 8),
        "shallow": _clip(1 - (item.get("retrace") or 0.5) / 0.6),
        "touches": _clip((item.get("touches", 4) - 4) / 8),
        "match": _clip((item.get("score", 0.8) - 0.8) / 0.2),
        "volume": c.volume(int(pole[0]), int(pole[2] - pole[0])),
    }


def _range(c, item):
    a = c.a
    bottom, top = item["box"]
    end = int(min(item.get("end", item["start"] + item["window"] - 1), a.n - 1))
    return {
        "touches": _clip((item.get("touches", 4) - 4) / 8),
        "length": _clip(item.get("window", 20) / 100),
        "poc": 1.0 if item.get("poc") is not None else 0.0,
        "height": _clip((top - bottom) / a.atr[end] / 6),
    }


GROUPS = {"zone": _zone, "structure": _structure, "eq": _eq, "shape": _shape,
          "figure": _figure, "flag": _flag, "range": _range}


def factors(df, item):
    """Критерии находки {критерий: 0..1} или None, если группа неизвестна."""
    group = group_of(item)
    if group is None:
        return None
    try:
        return {k: _clip(v) for k, v in GROUPS[group](context(df), item).items()}
    except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError):
        return None


def expert_logit(group, facts, weights=None, bias=None):
    """Сила «по учебнику» в виде логита: веса × критерии, 0 — середина."""
    weights = weights or PRIOR[group]
    total = sum(PRIOR[group].values())
    score = sum(weights.get(k, 0) * facts.get(k, 0.5) for k in PRIOR[group]) / total
    return SCALE * score + (-SCALE * 0.5 if bias is None else bias)


def level_of(logit):
    if logit >= SCALE * (STRONG - 0.5):
        return "strong"
    if logit < SCALE * (WEAK - 0.5):
        return "weak"
    return "normal"


def evaluate(df, item, personal=None):
    """Оценка находки: пишет item["strength"] (0..1), item["grade"] (strong / normal / weak)
    и item["factors"]. personal — личная модель пользователя (feedback.Personal) или None."""
    if "strength" in item and personal is None:
        return item["strength"]
    group = group_of(item)
    facts = factors(df, item)
    if group is None or facts is None:
        return None
    mine = personal is not None and personal.group == group
    logit = personal.logit(facts) if mine else expert_logit(group, facts)
    family = quality.family_of(item)
    if family and quality.MODELS.get(family, {}).get("on"):            # значимость по истории рынка
        p = quality.score(df, item)
        base = (quality.MODELS[family].get("exam") or {}).get("base", 0.4)
        if p is not None:
            logit += QUALITY_WEIGHT * (p - base)
    item["factors"] = facts
    item["group"] = group
    item["strength"] = float(1 / (1 + math.exp(-logit)))
    item["grade"] = level_of(logit)
    return item["strength"]


def passes(item, settings=None):
    """Показывать ли: «обычная» строгость прячет слабые, «строгая» — всё, кроме сильных."""
    level = item.get("grade")
    mode = quality.strictness(settings)
    if level is None or mode == "off":
        return True
    return level == "strong" if mode == "strict" else level != "weak"


def chooser(df, similar=None):
    """Ключ выбора лучшего кандидата: сила, а с твоими примерами — похожесть и сила вместе."""
    def key(item):
        s = evaluate(df, item)
        sim = similar(item) if similar else None
        if s is None:
            return sim or 0
        return 0.6 * sim + 0.4 * s if sim is not None else s
    return key


def strongest(items, n=3):
    """Самые весомые критерии находки — словами (для «📚 Обучение» и подсказок)."""
    facts = items.get("factors") or {}
    group = items.get("group")
    if not facts or group not in PRIOR:
        return []
    ranked = sorted(facts, key=lambda k: facts[k] * PRIOR[group].get(k, 0), reverse=True)
    return [FACTOR_NAMES[k] for k in ranked[:n] if facts[k] >= 0.6]


def nudge(item, delta):
    """Сдвинуть силу находки (в логитах) — например, по твоим 👍/👎 похожим находкам."""
    s = min(max(item["strength"], 1e-6), 1 - 1e-6)
    logit = math.log(s / (1 - s)) + delta
    item["strength"] = float(1 / (1 + math.exp(-logit)))
    item["grade"] = level_of(logit)
