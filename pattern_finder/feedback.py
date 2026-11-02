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

Это машинное обучение в самой простой и понятной форме: модель — это пороги и
список оценённых примеров, её можно прочитать глазами («📚 Обучение» в боте).
"""

from structure import structure_events

MIN_VOTES_RULES = 8        # правила — от 8 оценок паттерна (и хотя бы по одной 👍 и 👎)
MIN_VOTES_FIT = 5          # «подходит тебе N%» — от 5 оценок
KEEP_LIKES = 0.9           # правило должно сохранять ≥ 90% понравившихся находок
NEIGHBOURS = 7
MAX_RULES = 2

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
    elif item.get("kind") in ("eqh", "eql"):
        f["distance_pct"] = item.get("distance", 0) * 100
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


def rule_text(rule):
    label, fmt = FEATURES.get(rule["feature"], (rule["feature"], "{:g}"))
    if fmt == "bool":
        want_yes = (rule["op"] == ">=" and rule["value"] >= 1) or (rule["op"] == "<=" and rule["value"] < 1)
        return f"только {label}" if want_yes else f"не {label}"
    sign = "≥" if rule["op"] == ">=" else "≤"
    return f"{label} {sign} {fmt.format(rule['value'])}"


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


# --- всё вместе ---------------------------------------------------------------------------------------
def learn(feedback):
    """Модель пользователя по всем оценкам: {паттерн: {"records", "rules"}}."""
    model = {}
    for record in feedback:
        model.setdefault(record["name"], {"records": [], "rules": []})["records"].append(record)
    for entry in model.values():
        entry["rules"] = learn_rules(entry["records"])
    return model


def make_judge(feedback, use_rules=True):
    """Функция для сканера: (паттерн, df, находка) → (присылать?, «подходит тебе» 0..1 или None)."""
    model = learn(feedback)

    def judge(name, df, item):
        entry = model.get(name)
        if entry is None:
            return True, None
        feats = features(df, item)
        if use_rules and not all(passes(feats, rule) for rule in entry["rules"]):
            return False, None
        return True, fit_probability(entry["records"], feats)
    return judge


def summary(feedback):
    """Текст для «📚 Обучение»: оценки и выученные правила по каждому паттерну."""
    model = learn(feedback)
    if not model:
        return ("Пока нет ни одной оценки. Ставь 👍 / 👎 под алертами — после "
                f"{MIN_VOTES_FIT} оценок паттерна я начну писать «подходит тебе N%», "
                f"после {MIN_VOTES_RULES} — сам отсеивать то, что тебе не нравится.")
    lines = ["📚 Чему я научился по твоим оценкам:"]
    for name, entry in sorted(model.items()):
        records = entry["records"]
        likes = sum(r["vote"] > 0 for r in records)
        line = f"\n• {name}: 👍 {likes}, 👎 {len(records) - likes}"
        if entry["rules"]:
            line += "\n   присылаю только если: " + "; ".join(rule_text(r) for r in entry["rules"])
        elif len(records) < MIN_VOTES_RULES:
            line += f"\n   правила — после {MIN_VOTES_RULES} оценок (ещё {MIN_VOTES_RULES - len(records)})"
        else:
            line += "\n   чёткой границы между 👍 и 👎 пока нет — продолжай оценивать"
        lines.append(line)
    return "\n".join(lines)
