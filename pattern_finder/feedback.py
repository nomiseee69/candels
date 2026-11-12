"""Обучение на оценках пользователя: 👍 «подходит» / 👎 «не то».

Каждая находка описывается набором чисел — «признаками»: сколько касаний у фигуры,
насколько похожа на пример, какого размера зона, далеко ли цена, по тренду ли.
Оценки пользователя превращаются в два инструмента:

1) ПРАВИЛА («пороги»). Для каждого паттерна ищем признак и порог, которые отделяют
   то, что нравится, от того, что не нравится. Например: из 12 оценок «Канала»
   все 👍 — от 6 касаний, а 👎 — в основном 4–5 касаний → правило «касаний ≥ 6».
   Правило принимаем, только если оно сохраняет почти все 👍 (≥ 90%) и отсекает
   заметную часть 👎. Такие находки бот больше не присылает.

2) «ПОДХОДИТ ТЕБЕ N%». Для новой находки берём самые похожие по признакам находки,
   которые пользователь уже оценил (метод ближайших соседей), и считаем долю 👍 среди них.

3) ЛИЧНАЯ МОДЕЛЬ ПО ВИДУ (дообучение на 👍/👎). Вместе с оценкой бот запоминает, как
   находка выглядела (участок свечей). «Глаз» (HOG или своя нейросеть) превращает каждый
   участок в набор чисел, и поверх них учится маленькая логистическая регрессия:
   «такой вид тебе нравится / не нравится». Сама нейросеть при этом не меняется —
   переучивается только её «последний слой» под тебя. Так делают и большие системы,
   когда данных у одного человека мало (десятки оценок, а не тысячи).

4) ЛИЧНЫЕ ВЕСА КРИТЕРИЕВ (главное). Эксперт (strength.py) оценивает силу находки по
   критериям: импульс, снятие ликвидности, объём, дисконт, тренд, свежесть… — с весами
   «по учебнику». С каждой 👍/👎 бот подстраивает ЭТИ веса под тебя: если ты ставишь 👎
   находкам без снятия ликвидности, вес «снятие ликвидности» растёт; если 👍 и слабым на вид —
   строгость отбора для тебя снижается. Чем меньше оценок, тем ближе веса к учебнику
   (бот не делает выводов по двум кликам).

5) УСЛОВИЯ СЛОВАМИ ИЗ ОЦЕНОК. С каждой оценкой бот запоминает, какие понятия у находки
   были (снятие ликвидности, объём, FVG внутри, свежесть, ключевой слом…, language.present).
   Понятие, которое есть почти у всех 👍 и редко у 👎, становится «обязательно»; наоборот —
   «нельзя». Это то же самое, как если бы ты описал паттерн словами, только выучено само.
   Выученное объединяется с твоим описанием (language.merge) и сразу работает в поиске.

6) 👍 = ЕЩЁ ОДИН ПРИМЕР. Отпечаток понравившейся находки становится примером паттерна:
   дальше ищу и похожие на неё. 👎 — эта находка больше не придёт никогда.

Чему бот научился — словами в «📚 Обучение» (без чисел).
"""

import numpy as np

import language
import strength
import visual
from structure import structure_events

MIN_VOTES_RULES = 8        # правила — от 8 оценок паттерна (и хотя бы по одной 👍 и 👎)
MIN_VOTES_FIT = 5          # «подходит тебе N%» — от 5 оценок
KEEP_LIKES = 0.9           # правило должно сохранять ≥ 90% понравившихся находок
NEIGHBOURS = 7
MAX_RULES = 2
MIN_VOTES_HEAD = 10        # личная модель по виду — от 10 оценок паттерна…
MIN_EACH_HEAD = 3          # …среди них хотя бы 3 👍 и 3 👎
WINDOW_POINTS = 32         # как выглядела находка: 32 точки open/high/low/close
MIN_VOTES_ADAPT = 3        # личные веса критериев — от 3 оценок паттерна
PRIOR_STRENGTH = 0.5       # насколько держимся за «учебник» (больше — нужны более уверенные оценки)
FIT_WEIGHT = 3.0           # насколько «похоже на то, что ты лайкал» сдвигает силу находки

MAX_VOTE_EXAMPLES = 8      # 👍 → пример: не больше 8 последних на паттерн
MIN_LIKES_SPEC = 3         # «обязательно» из оценок — от 3 👍 и 2 👎
MIN_DISLIKES_SPEC = 3      # «нельзя» из оценок — от 3 👎 и 2 👍

FEATURES = {
    # ключ: (как назвать в тексте, формат значения)
    "touches": ("касаний", "{:.0f}"),
    "similarity": ("похожесть на пример", "{:.0%}"),
    "violations": ("нарушений линий", "{:.0f}"),
    "length": ("длина, свечей", "{:.0f}"),
    "height_pct": ("высота фигуры", "{:.1f}%"),
    "size_pct": ("размер зоны", "{:.2f}%"),
    "distance_pct": ("расстояние до цены", "{:.1f}%"),
    "age": ("возраст, свечей", "{:.0f}"),
    "score": ("оценка совпадения", "{:.0%}"),
    "with_trend": ("по тренду", "bool"),
    "significant": ("значимый", "bool"),
    "imbalance": ("с FVG внутри", "bool"),
    "tested": ("цена уже возвращалась", "bool"),
    "volume_ratio": ("объём блока, × среднего", "{:.1f}"),
}


def _direction(item):
    """Куда «смотрит» находка: "up", "down" или None (у фигур направление не задано)."""
    kind = str(item.get("kind", ""))
    if kind.startswith("bullish") or item.get("direction") == "up" or kind == "eql":
        return "up"
    if kind.startswith("bearish") or item.get("direction") == "down" or kind == "eqh":
        return "down"
    return None


def features(df, item):
    """Признаки находки: словарь {признак: число}. Только то, что у неё есть."""
    price = float(df["close"].iloc[-1])
    end = min(item["start"] + item["window"] - 1, len(df) - 1)
    f = {"score": item.get("score"), "age": len(df) - 1 - end}
    if "upper" in item:                                           # треугольник, клин, канал
        f.update(touches=item.get("touches"), violations=item.get("violations"),
                 similarity=item.get("similarity"), length=item.get("window"))
        height = item["upper"][1] - item["lower"][1]
        f["height_pct"] = abs(height) / price * 100 if price else None
    elif "zone" in item:                                          # FVG и блоки
        bottom, top = item["zone"]
        f["size_pct"] = (top - bottom) / price * 100 if price else None
        f["distance_pct"] = item.get("distance", 0) * 100
        if "significant" in item:
            f["significant"] = int(bool(item["significant"]))
        if "imbalance" in item:
            f["imbalance"] = int(bool(item["imbalance"]))
        if "touched_at" in item:
            f["tested"] = int(item["touched_at"] is not None)
        if item.get("volume_ratio") is not None:
            f["volume_ratio"] = item["volume_ratio"]
    elif item.get("kind") in ("eqh", "eql"):
        f["distance_pct"] = item.get("distance", 0) * 100
    elif item.get("kind") == "range":                                # боковик
        bottom, top = item["box"]
        f.update(touches=item.get("touches"), length=item.get("window"),
                 height_pct=(top - bottom) / price * 100 if price else None,
                 distance_pct=item.get("distance", 0) * 100)
    if item.get("similarity") is not None:
        f["similarity"] = item["similarity"]
    direction = _direction(item)
    if direction is not None:
        events = structure_events(df)
        if events:
            f["with_trend"] = int(events[-1]["direction"] == direction)
    return {k: float(v) for k, v in f.items() if v is not None}


# --- правила --------------------------------------------------------------------------------------
def passes(feats, rule):
    """Проходит ли находка правило. Нет такого признака — не отсекаем."""
    value = feats.get(rule["feature"])
    if value is None:
        return True
    return value >= rule["value"] if rule["op"] == ">=" else value <= rule["value"]


def learn_rules(records):
    """Пороги, которые отделяют 👍 от 👎 (не больше MAX_RULES). records — оценки одного паттерна."""
    likes = [r for r in records if r["vote"] > 0]
    dislikes = [r for r in records if r["vote"] < 0]
    if len(records) < MIN_VOTES_RULES or not likes or not dislikes:
        return []
    rules = []
    total_likes = len(likes)
    for _ in range(MAX_RULES):
        best = None
        names = {k for r in records for k in r["features"]}
        # порядок важен при равной пользе: понятные признаки (касания, похожесть) — первыми
        for name in sorted(names, key=lambda k: (list(FEATURES).index(k) if k in FEATURES else 99, k)):
            values = sorted({r["features"][name] for r in records if name in r["features"]})
            for value in values:
                for op in (">=", "<="):
                    rule = {"feature": name, "op": op, "value": value}
                    kept = [r for r in likes if passes(r["features"], rule)]
                    cut = [r for r in dislikes if not passes(r["features"], rule)]
                    # все правила вместе должны сохранить ≥ 90% всех 👍
                    if len(kept) < KEEP_LIKES * total_likes or len(cut) < max(2, 0.3 * len(dislikes)):
                        continue
                    gain = len(cut) - 2 * (len(likes) - len(kept))
                    if best is None or gain > best[0]:
                        best = (gain, rule, kept, cut)
        if best is None:
            break
        _, rule, likes, cut = best
        rules.append(rule)
        dislikes = [r for r in dislikes if r not in cut]
        if not dislikes:
            break
    return rules


RULE_WORDS = {   # правило словами, без чисел: (если «≥», если «≤»)
    "touches": ("много касаний", "мало касаний"),
    "similarity": ("очень похожие на твой пример", "не слишком похожие на пример"),
    "violations": ("с выбросами за линии", "без выбросов за линии"),
    "length": ("длинные фигуры", "короткие фигуры"),
    "height_pct": ("крупные фигуры", "небольшие фигуры"),
    "size_pct": ("крупные зоны", "узкие зоны"),
    "distance_pct": ("далёкие от цены", "близкие к цене"),
    "age": ("давние", "свежие"),
    "score": ("с точной формой", "с неточной формой"),
    "volume_ratio": ("с повышенным объёмом", "без всплеска объёма"),
}


def rule_text(rule):
    """Правило словами — как ты его «сказал» оценками. Чисел не показываем."""
    label, fmt = FEATURES.get(rule["feature"], (rule["feature"], "{:g}"))
    if fmt == "bool":
        want_yes = (rule["op"] == ">=" and rule["value"] >= 1) or (rule["op"] == "<=" and rule["value"] < 1)
        return f"только {label}" if want_yes else f"не {label}"
    more, less = RULE_WORDS.get(rule["feature"], (f"{label}: больше", f"{label}: меньше"))
    return more if rule["op"] == ">=" else less


# --- «подходит тебе N%» ------------------------------------------------------------------------------
def fit_probability(records, feats):
    """Доля 👍 среди NEIGHBOURS самых похожих оценённых находок (0..1) или None."""
    if len(records) < MIN_VOTES_FIT:
        return None
    keys = [k for k in feats if any(k in r["features"] for r in records)]
    if not keys:
        return None
    spans = {}
    for k in keys:
        values = [r["features"][k] for r in records if k in r["features"]] + [feats[k]]
        spans[k] = (max(values) - min(values)) or 1.0

    def distance(r):
        diffs = [abs(r["features"][k] - feats[k]) / spans[k] for k in keys if k in r["features"]]
        return sum(diffs) / len(diffs) if diffs else 1.0

    nearest = sorted(records, key=distance)[:NEIGHBOURS]
    weights = [1.0 / (0.05 + distance(r)) for r in nearest]           # ближе — важнее
    likes = sum(w for w, r in zip(weights, nearest) if r["vote"] > 0)
    return likes / sum(weights)


# --- личная модель по виду -------------------------------------------------------------------------
def item_window(df, item, before=40, after=20, points=WINDOW_POINTS):
    """Как выглядела находка: участок свечей вокруг неё, 0..1, медвежьи перевёрнуты в бычьи,
    сжат до 32 точек — коротко, чтобы хранить в users.json вместе с оценкой."""
    n = len(df)
    start = int(item["start"])
    end = min(int(item.get("end", start + item.get("window", 1) - 1)), n - 1)
    kind = str(item.get("kind", ""))
    bearish = kind.startswith("bearish") or item.get("direction") == "down" or kind == "eqh"
    ohlc = visual.normalize_ohlc(df, max(0, start - before), min(n, end + 1 + after), mirror=bearish)
    if ohlc is None or len(ohlc) < 5:
        return None
    x_old, x_new = np.linspace(0, 1, len(ohlc)), np.linspace(0, 1, points)
    small = np.stack([np.interp(x_new, x_old, ohlc[:, k]) for k in range(4)], axis=1)
    return np.round(small, 3).ravel().tolist()


def _look(window):
    return visual.look(np.asarray(window, dtype=float).reshape(-1, 4))


def personal_head(records, steps=300, rate=0.5, l2=0.01):
    """Логистическая регрессия «нравится ли тебе такой вид» поверх описаний глаза.
    Возвращает функцию (окно) → вероятность 👍 или None, если оценок мало."""
    rows = [(r["window"], r["vote"] > 0) for r in records if r.get("window")]
    likes = sum(y for _, y in rows)
    if len(rows) < MIN_VOTES_HEAD or likes < MIN_EACH_HEAD or len(rows) - likes < MIN_EACH_HEAD:
        return None
    X = np.array([_look(w) for w, _ in rows])
    y = np.array([float(v) for _, v in rows])
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-6
    X = (X - mean) / scale
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):                       # градиентный спуск: подвинуть веса туда, где ошибка меньше
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= rate * (X.T @ (p - y) / len(y) + l2 * w)
        b -= rate * float(np.mean(p - y))

    def predict(window):
        x = (_look(window) - mean) / scale
        return float(1 / (1 + np.exp(-(x @ w + b))))
    predict.votes = len(rows)
    return predict


# --- личные веса критериев силы ------------------------------------------------------------------------
class Personal:
    """Твоя версия эксперта: те же критерии, но веса и строгость подстроены по 👍/👎."""

    def __init__(self, group, weights, bias, votes):
        self.group, self.weights, self.bias, self.votes = group, weights, bias, votes

    def logit(self, facts):
        return strength.expert_logit(self.group, facts, self.weights, self.bias)

    def changes(self):
        """Что изменилось относительно «учебника» — словами: (важнее, менее важно, строгость)."""
        prior = strength.PRIOR[self.group]
        more = [strength.FACTOR_NAMES[k] for k in prior if self.weights[k] >= prior[k] * 1.3 + 0.1]
        less = [strength.FACTOR_NAMES[k] for k in prior if self.weights[k] <= prior[k] * 0.7 - 0.1]
        shift = self.bias + strength.SCALE * 0.5
        mood = "показываю тебе больше находок" if shift > 0.5 else (
            "отбираю для тебя строже" if shift < -0.5 else "")
        return more, less, mood


def personal_weights(records, steps=400):
    """Подстройка весов эксперта по оценкам одного паттерна (логистическая регрессия,
    которая «держится» за учебные веса: штраф за отход от них). None — оценок мало."""
    rows = [(r["factors"], r["vote"] > 0, r.get("group")) for r in records if r.get("factors")]
    if len(rows) < MIN_VOTES_ADAPT:
        return None
    group = max({g for _, _, g in rows if g}, key=lambda g: sum(1 for *_, x in rows if x == g), default=None)
    if group not in strength.PRIOR:
        return None
    keys = list(strength.PRIOR[group])
    rows = [(f, y) for f, y, g in rows if g == group]
    X = np.array([[f.get(k, 0.5) for k in keys] for f, _ in rows])
    y = np.array([float(v) for _, v in rows])
    scale = strength.SCALE / sum(strength.PRIOR[group].values())
    w0 = np.array([strength.PRIOR[group][k] for k in keys])
    b0 = -strength.SCALE * 0.5
    w, b = w0.copy(), b0
    rate = 1.0 / (scale ** 2 * len(rows) / 4 + 2 * PRIOR_STRENGTH)      # шаг, при котором спуск не «раскачивается»
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(scale * X @ w + b)))
        w -= rate * (scale * X.T @ (p - y) + 2 * PRIOR_STRENGTH * (w - w0))
        b -= rate * (float(np.sum(p - y)) + 2 * PRIOR_STRENGTH * (b - b0))
        w = np.maximum(w, 0)                    # вес не бывает отрицательным: критерий либо помогает, либо нет
    return Personal(group, dict(zip(keys, w.tolist())), float(b), len(rows))


# --- условия словами из оценок -----------------------------------------------------------------------
def learned_spec(records):
    """Условия, выученные по 👍/👎 (как описание словами): {"require", "prefer", "forbid",
    "direction", "timeframes"} или None, если отличий пока не видно."""
    likes = [r for r in records if r["vote"] > 0 and r.get("facts") is not None]
    dislikes = [r for r in records if r["vote"] < 0 and r.get("facts") is not None]
    spec = {"require": [], "prefer": [], "forbid": [], "timeframes": [], "direction": None,
            "min_touches": None, "learned": True}
    keys = sorted({k for r in likes + dislikes for k in r["facts"]})
    for key in keys:
        if key not in language.CONCEPTS:
            continue
        a = [r["facts"][key] for r in likes if key in r["facts"]]
        b = [r["facts"][key] for r in dislikes if key in r["facts"]]
        pa = sum(a) / len(a) if a else None
        pb = sum(b) / len(b) if b else None
        if len(a) >= MIN_LIKES_SPEC and len(b) >= 2 and pa >= 0.8 and pb <= 0.4:
            spec["require"].append(key)
        elif len(b) >= MIN_DISLIKES_SPEC and len(a) >= 2 and pb >= 0.8 and pa <= 0.2:
            spec["forbid"].append(key)
        elif len(a) >= 2 and len(b) >= 1 and pa - pb >= 0.5:
            spec["prefer"].append(key)
    for mode in ("require", "prefer", "forbid"):
        if "imbalance" in spec[mode] and "inside_ob" in spec[mode]:
            spec[mode].remove("inside_ob")                   # одно и то же условие
    ups = [r.get("direction") for r in likes if r.get("direction")]
    downs = [r.get("direction") for r in dislikes if r.get("direction")]
    for d in ("up", "down"):
        if len(ups) >= MIN_LIKES_SPEC and all(x == d for x in ups) and len(downs) >= 2 \
                and sum(x != d for x in downs) >= 0.8 * len(downs):
            spec["direction"] = d
    tfs_liked = {r.get("interval") for r in likes}
    tfs_disliked = [r.get("interval") for r in dislikes]
    if len(likes) >= MIN_LIKES_SPEC and len(tfs_disliked) >= MIN_DISLIKES_SPEC \
            and not any(tf in tfs_liked for tf in tfs_disliked) and None not in tfs_liked:
        spec["timeframes"] = sorted(tfs_liked)
    return None if language.is_empty(spec) else spec


def vote_examples(feedback):
    """👍 → ещё один пример паттерна (отпечаток находки): [{"name", "fingerprint", "auto"}]."""
    out, per_name = [], {}
    for r in reversed(feedback):
        if r["vote"] > 0 and (r.get("fingerprint") or r.get("example")):
            n = per_name.get(r["name"], 0)
            if n < MAX_VOTE_EXAMPLES:
                per_name[r["name"]] = n + 1
                entry = {"name": r["name"], "kind": r.get("kind"), "auto": True, "note": "👍 с рынка"}
                if r.get("fingerprint"):
                    entry["fingerprint"] = r["fingerprint"]
                else:                                         # фигура: эталон линий (reference) / форма (shape)
                    entry.update(r["example"])
                out.append(entry)
    return out


def with_learning(patterns, feedback):
    """Примеры пользователя + чему бот научился по оценкам — то, с чем идёт поиск:
    к описанию каждого примера добавляются выученные условия, 👍 — новые примеры-отпечатки."""
    if not feedback:
        return patterns
    model = learn(feedback)
    result = []
    for p in patterns:
        spec = (model.get(p["name"]) or {}).get("spec")
        if spec:
            p = {**p, "spec": language.merge(p.get("spec"), spec)}
        result.append(p)
    names = {p["name"] for p in patterns}
    return result + [e for e in vote_examples(feedback) if e["name"] in names]


def disliked_keys(feedback):
    return {r["key"] for r in feedback if r["vote"] < 0 and r.get("key")}


def what_changed(before, after, vote):
    """Что изменилось после оценки — словами для ответа пользователю."""
    lines = []
    bs, as_ = (before or {}).get("spec") or {}, (after or {}).get("spec") or {}
    for mode, title in (("require", "теперь обязательно"), ("forbid", "теперь нельзя"),
                        ("prefer", "теперь желательно")):
        new = [k for k in as_.get(mode) or [] if k not in (bs.get(mode) or [])]
        if new:
            lines.append(f"{title}: " + "; ".join(language.CONCEPTS[k][0] for k in new))
    if as_.get("direction") and as_["direction"] != bs.get("direction"):
        lines.append("теперь присылаю только " + ("бычьи ↑" if as_["direction"] == "up" else "медвежьи ↓"))
    if as_.get("timeframes") and as_["timeframes"] != bs.get("timeframes"):
        lines.append("теперь только на " + ", ".join(t.upper() for t in as_["timeframes"]))
    bp, ap = (before or {}).get("personal"), (after or {}).get("personal")
    if ap is not None and ap.votes >= 5:                 # по паре оценок про веса не рассказываем
        more, less, mood = ap.changes()
        bmore, bless, bmood = bp.changes() if bp is not None and bp.votes >= 5 else ([], [], "")
        if [m for m in more if m not in bmore]:
            lines.append("важнее для тебя: " + ", ".join(m for m in more if m not in bmore))
        if [m for m in less if m not in bless]:
            lines.append("менее важно: " + ", ".join(m for m in less if m not in bless))
        if mood and mood != bmood:
            lines.append(mood)
    return lines


# --- всё вместе ---------------------------------------------------------------------------------------
def learn(feedback):
    """Модель пользователя по всем оценкам: {паттерн: {"records", "rules"}}."""
    model = {}
    for record in feedback:
        model.setdefault(record["name"], {"records": [], "rules": []})["records"].append(record)
    for entry in model.values():
        entry["rules"] = learn_rules(entry["records"])
        entry["head"] = personal_head(entry["records"])
        entry["personal"] = personal_weights(entry["records"])
        entry["spec"] = learned_spec(entry["records"])
    return model


def make_judge(feedback, use_rules=True):
    """Функция для сканера: (паттерн, df, находка) → (присылать?, «подходит тебе» 0..1 или None).
    Заодно пересчитывает силу находки ЛИЧНЫМИ весами (strength / level в находке):
    дальше сканер отбирает по ней."""
    model = learn(feedback)

    def judge(name, df, item):
        entry = model.get(name)
        if entry is None:
            return True, None
        feats = features(df, item)
        if use_rules and not all(passes(feats, rule) for rule in entry["rules"]):
            return False, None
        guesses = [fit_probability(entry["records"], feats)]
        if entry.get("head"):
            window = item_window(df, item)
            if window is not None:
                guesses.append(entry["head"](window))         # личная модель по виду
        guesses = [g for g in guesses if g is not None]
        fit = sum(guesses) / len(guesses) if guesses else None
        if entry.get("personal") is not None:
            strength.evaluate(df, item, entry["personal"])    # твои веса критериев и строгость
        if fit is not None and item.get("strength") is not None:
            strength.nudge(item, FIT_WEIGHT * (fit - 0.5))    # похоже на то, что ты лайкал
        return True, fit
    judge.disliked = disliked_keys(feedback)            # 👎 — эту находку больше не присылать
    return judge


def summary(feedback):
    """Текст для «📚 Обучение»: чему бот научился по каждому паттерну — словами, без чисел."""
    model = learn(feedback)
    if not model:
        return ("Пока нет ни одной оценки. Ставь 👍 / 👎 под алертами — с первых оценок "
                "я начну подстраивать под тебя, какие критерии паттерна важнее и насколько "
                "строго отбирать находки.")
    lines = ["Чему я научился по твоим оценкам:"]
    for name, entry in sorted(model.items()):
        records = entry["records"]
        likes = sum(r["vote"] > 0 for r in records)
        line = f"\n• {name}: 👍 {likes}, 👎 {len(records) - likes}"
        personal = entry.get("personal")
        if personal is not None:
            more, less, mood = personal.changes()
            if more:
                line += "\n   для тебя важнее: " + ", ".join(more)
            if less:
                line += "\n   для тебя менее важно: " + ", ".join(less)
            if mood:
                line += f"\n   {mood}"
            if not (more or less or mood):
                line += "\n   пока твой вкус совпадает с учебником — продолжай оценивать"
        else:
            line += "\n   подстраиваться начну после пары оценок"
        if entry.get("spec"):
            line += "\n   выучил условия: " + "; ".join(language.describe(entry["spec"]))
        if any(r["vote"] > 0 and (r.get("fingerprint") or r.get("example")) for r in records):
            line += "\n   понравившиеся находки — мои новые примеры, ищу похожие"
        if entry["rules"]:
            line += "\n   присылаю только: " + "; ".join(rule_text(r) for r in entry["rules"])
        if entry.get("head"):
            line += "\n   запомнил, как выглядят находки, которые тебе нравятся"
        lines.append(line)
    return "\n".join(lines)
