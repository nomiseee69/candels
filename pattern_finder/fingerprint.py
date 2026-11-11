"""«Отпечаток» примера пользователя для SMC, Range, флагов — и сравнение находок с ним.

Главная идея проекта: бот ищет то, что похоже на ТВОИ примеры. Для фигур (треугольник,
двойное дно) это давно работает по форме. Для Order Block, FVG, BOS, Range правило
одно для всех — а вот КАКОЙ блок или боковик тебе нравится, у каждого своё:
кому-то — маленький блок с огромным импульсом и объёмом, кому-то — широкий,
от которого цена ушла медленно.

Поэтому со скриншота бот:
  1) читает свечи (vision.extract_candles) — в пикселях, но это не важно;
  2) находит на них твой паттерн тем же правилом, что и на рынке;
  3) снимает отпечаток — числа, которые не зависят от масштаба (всё в «средних свечах» ATR):
       Order Block — высота зоны, сила импульса, объём ×N, есть ли FVG внутри;
       FVG — размер дыры, сила импульсной свечи, объём;
       BOS / CHoCH — насколько далеко закрылась свеча за уровнем, длина хода;
       Range — высота, длина, касания, где POC;
       флаг / вымпел — флагшток, откат, длина флага;
     и вид участка (visual.py) — как он выглядит картинкой.
На рынке каждая находка получает такой же отпечаток и оценку «похоже на твой пример» (внутри — число):
65% — числа, 35% — вид. Из нескольких находок на монете выбирается самая похожая.
"""

import numpy as np

from blocks import all_blocks, volume_ratios
from flags import find_flag, history_flags
from ranges import EDGE, find_range, history_ranges, volume_profile
from trendlines import pivot_highs, pivot_lows
from smc import find_all_fvg
from structure import atr, equal_levels, structure_events
import visual
from blocks import BLOCK_NAMES
from flags import NAMES as FLAG_NAMES
from smc import FVG_NAMES
from structure import EQ_NAMES

FAMILY = {
    "Order Block": "block", "Breaker Block": "block", "Mitigation Block": "block",
    "FVG": "fvg", "BOS": "structure", "CHoCH": "structure", "EQH / EQL": "eq",
    "Range": "range", "Флаг": "flags", "Вымпел": "flags",
}
BLOCK = {"Order Block": "ob", "Breaker Block": "breaker", "Mitigation Block": "mitigation"}
STRUCTURE = {"BOS": "bos", "CHoCH": "choch"}
FLAG = {"Флаг": "flag", "Вымпел": "pennant"}

# как сравнивать признак: "size" — во сколько раз отличается (вдвое = совсем не похоже),
# "share" — доли 0..1, "flag" — да/нет
FEATURE_KIND = {
    "zone_atr": "size", "impulse_atr": "size", "volume_ratio": "size", "imbalance": "flag",
    "bars": "size", "span": "size", "gap_atr": "size", "body_atr": "size", "significant": "flag",
    "break_atr": "size", "leg_atr": "size", "spread_atr": "size", "touches": "size",
    "height_atr": "size", "length": "size", "poc_rel": "share", "pole_atr": "size",
    "retrace": "share", "body_ratio": "size",
}
FEATURE_TEXT = {
    "zone_atr": "зона {:.1f} ср. свечи", "impulse_atr": "импульс {:.1f} ср. свечи",
    "volume_ratio": "объём ×{:.1f}", "imbalance": "FVG внутри: {}", "bars": "{:.0f} свечей до слома", "span": "{:.0f} свечей между касаниями",
    "gap_atr": "дыра {:.1f} ср. свечи", "body_atr": "импульсная свеча {:.1f} ср. свечи",
    "significant": "значимый: {}", "break_atr": "закрытие за уровнем на {:.1f} ср. свечи",
    "leg_atr": "ход {:.1f} ср. свечи", "spread_atr": "разброс {:.2f} ср. свечи", "touches": "касаний {:.0f}",
    "height_atr": "высота {:.1f} ср. свечи", "length": "{:.0f} свечей", "poc_rel": "POC на {:.0%} высоты",
    "pole_atr": "флагшток {:.1f} ср. свечи", "retrace": "откат {:.0%}", "body_ratio": "флаг {:.1f}× флагштока",
}
FEATURE_WEIGHT = 0.65          # итог: 65% — числа, 35% — вид
CONTEXT_BEFORE = 40            # вид сравниваем на участке: 40 свечей до паттерна…
CONTEXT_AFTER = 20             # …и 20 после
DRAWN_BONUS = 3.0              # паттерн там, где ты нарисовал линию, — точно он


# --- признаки одной находки --------------------------------------------------------------------
def _end(item, n):
    return min(int(item.get("end", item["start"] + item.get("window", 1) - 1)), n - 1)


def _size(df, item):
    """Средняя свеча (ATR) в момент паттерна — единица измерения всех признаков."""
    sizes = atr(df).to_numpy()
    value = float(np.mean(sizes[item["start"]:_end(item, len(df)) + 1]))
    return value if value > 0 else float(np.mean(sizes)) or 1.0


def features_of(family, df, item):
    """Отпечаток находки: {признак: число}. Масштаб графика не важен — всё в ATR и долях."""
    n = len(df)
    start, end = int(item["start"]), _end(item, n)
    size = _size(df, item)
    highs, lows = df["high"].to_numpy(float), df["low"].to_numpy(float)
    f = {}
    if family == "block":
        bottom, top = item["zone"]
        f["zone_atr"] = (top - bottom) / size
        f["impulse_atr"] = (highs[start:end + 1].max() - lows[start:end + 1].min()) / size
        f["bars"] = max(1, end - start)
        f["imbalance"] = float(bool(item.get("imbalance")))
        if item.get("volume_ratio") is not None and item["volume_ratio"] > 0.05:   # 0 — объём не прочитан
            f["volume_ratio"] = item["volume_ratio"]
    elif family == "fvg":
        bottom, top = item["zone"]
        middle = min(start + 1, n - 1)
        f["gap_atr"] = (top - bottom) / size
        f["body_atr"] = abs(float(df["close"].iloc[middle]) - float(df["open"].iloc[middle])) / size
        f["significant"] = float(bool(item.get("significant")))
        ratios = volume_ratios(df)
        if ratios is not None and np.isfinite(ratios[middle]) and ratios[middle] > 0.05:
            f["volume_ratio"] = float(ratios[middle])
    elif family == "structure":
        f["break_atr"] = abs(float(df["close"].iloc[end]) - item["level"]) / size
        f["leg_atr"] = (highs[start:end + 1].max() - lows[start:end + 1].min()) / size
        f["bars"] = max(1, end - start)
    elif family == "eq":
        values = highs if item["kind"] == "eqh" else lows
        touched = [values[t] for t in item["touches"] if t < n]
        f["spread_atr"] = max(0.01, (max(touched) - min(touched)) / size)
        f["touches"] = len(touched)
        f["span"] = max(1, end - start)
    elif family == "range":
        bottom, top = item["box"]
        f["height_atr"] = (top - bottom) / size
        f["length"] = end - start + 1
        f["touches"] = item["touches"]
        if item.get("poc") is not None:
            f["poc_rel"] = (item["poc"] - bottom) / (top - bottom)
    elif family == "flags":
        x0, y0, x1, y1 = item["pole"]
        f["pole_atr"] = abs(y1 - y0) / size
        f["retrace"] = item["retrace"]
        f["body_ratio"] = max(1, end - x1) / max(1, x1 - x0)
        f["touches"] = item["touches"]
    return {k: float(v) for k, v in f.items()}


def feature_similarity(a, b):
    """Похожесть двух отпечатков 0..1 по общим признакам."""
    keys = [k for k in a if k in b]
    if not keys:
        return None
    distances = []
    for k in keys:
        kind = FEATURE_KIND.get(k, "size")
        if kind == "flag":
            distances.append(abs(a[k] - b[k]))
        elif kind == "share":
            distances.append(min(1.0, abs(a[k] - b[k]) / 0.5))
        else:                                             # вдвое больше / меньше — расстояние 1
            ratio = (abs(a[k]) + 0.05) / (abs(b[k]) + 0.05)
            distances.append(min(1.0, abs(np.log2(ratio))))
    return float(1 - np.mean(distances))


def is_bullish(item):
    kind = str(item.get("kind", ""))
    return kind.startswith("bullish") or item.get("direction") == "up" or kind == "eql"


# --- все случаи паттерна на графике (для скрина пользователя) -----------------------------------
def all_items(name, df):
    family = FAMILY[name]
    if family == "block":
        return [b for b in all_blocks(df) if b["family"] == BLOCK[name]]
    if family == "fvg":
        return find_all_fvg(df, lookback=len(df), include_filled=True)
    if family == "structure":
        return [e for e in structure_events(df) if e["kind"] == STRUCTURE[name]]
    if family == "eq":
        return equal_levels(df)
    if family == "range":
        now = find_range(df)
        return history_ranges(df) + ([now] if now else [])
    if family == "flags":
        now = find_flag(df, FLAG[name])
        return history_flags(df, FLAG[name]) + ([now] if now else [])
    return []


PROMINENCE = {
    "block": lambda f: f["impulse_atr"] + f.get("volume_ratio", 1),
    "fvg": lambda f: f["gap_atr"] + f["body_atr"],
    "structure": lambda f: f["break_atr"] + 0.3 * f["leg_atr"],
    "eq": lambda f: f["touches"],
    "range": lambda f: f["touches"] + f["length"] / 30,
    "flags": lambda f: f["pole_atr"],
}


def _price_range(item):
    if "zone" in item:
        return item["zone"]
    if "box" in item:
        return item["box"]
    if "level" in item:
        return item["level"], item["level"]
    return None


def _window(df, item):
    """Участок вокруг паттерна для сравнения «на вид»: свечи и где в нём паттерн."""
    n = len(df)
    start, end = int(item["start"]), _end(item, n)
    lo, hi = max(0, start - CONTEXT_BEFORE), min(n, end + 1 + CONTEXT_AFTER)
    ohlc = visual.normalize_ohlc(df, lo, hi)
    return None if ohlc is None else {"ohlc": np.round(ohlc, 4).tolist(), "start": start - lo, "end": end - lo}


def make_fingerprint(name, df, item):
    family = FAMILY[name]
    return {"family": family, "kind": item.get("kind"), "bullish": is_bullish(item),
            "features": features_of(family, df, item), "window": _window(df, item)}


ZONE_BONUS = 6.0               # паттерн совпал с зоной, которую ты закрасил, — это точно он
ZONE_MATCH = 0.3               # «совпал» — зоны перекрываются хотя бы на 30%


def _zone_match(item, zone):
    """Насколько находка совпадает с закрашенной зоной: 0..1 (перекрытие цен × близость по времени)."""
    prices = _price_range(item)
    if prices is None:
        return 0.0
    low, high = min(prices), max(prices)
    inter = min(high, zone["top"]) - max(low, zone["bottom"])
    union = max(high, zone["top"]) - min(low, zone["bottom"])
    overlap = max(0.0, inter) / union if union > 0 else 0.0
    distance = abs(int(item["start"]) - zone["start"])
    return overlap * (1.0 if distance <= 3 else 0.5 if distance <= 8 else 0.2)


def item_from_zone(name, df, zone):
    """Паттерн прямо по закрашенной зоне — когда правило на свечах со скрина его не нашло
    (например, после зоны на скрине мало свечей). Направление — куда ушла цена после зоны."""
    family = FAMILY[name]
    if family not in ("block", "fvg"):
        return None
    n = len(df)
    start = min(max(0, int(zone["start"])), n - 1)
    bottom, top = zone["bottom"], zone["top"]
    if family == "block":
        # свеча блока — та, чей размах лучше всего совпал с зоной; у OB это свеча ПРОТИВ движения:
        # перед падением — зелёная, перед ростом — красная
        after = df.iloc[start + 1:]
        falls = len(after) and bottom - float(after["low"].min()) > float(after["high"].max()) - top
        opens, closes = df["open"].to_numpy(float), df["close"].to_numpy(float)
        span = (top - bottom) or 1.0

        def fit(i):
            miss = (abs(float(df["high"].iloc[i]) - top) + abs(float(df["low"].iloc[i]) - bottom)) / span
            against = (closes[i] > opens[i]) if falls else (closes[i] < opens[i])
            return miss + (0 if against or BLOCK[name] != "ob" else 0.5)
        start = min(range(max(0, start - 2), min(n, start + 3)), key=fit)
    after = df.iloc[start + 1:]
    up = float(after["high"].max()) - top if len(after) else 0.0
    down = bottom - float(after["low"].min()) if len(after) else 0.0
    side = "bullish" if up >= down else "bearish"
    if family == "block" and len(after) >= 3:
        # блок — опора или сопротивление: где цена в основном была ПОСЛЕ зоны — над ней (бычий) или
        # под ней (медвежий). Надёжнее, чем один самый дальний выброс
        closes_after = after["close"].to_numpy(float)
        mid = (top + bottom) / 2
        above, below = float(np.mean(closes_after > mid)), float(np.mean(closes_after < mid))
        if abs(above - below) >= 0.3:
            side = "bullish" if above > below else "bearish"
        if BLOCK[name] == "breaker":
            # брейкер — блок, который ПРОБИЛИ: сначала цена ушла от него в одну сторону, потом вернулась и
            # закрылась с другой. Его сторона — та, где цена оказалась ПОСЛЕ пробоя: смотрим последнее
            # закрытие за пределами зоны (на длинном скрине «где цена была дольше» — это ещё время до пробоя)
            outside = np.flatnonzero((closes_after > top) | (closes_after < bottom))
            if len(outside):                           # последнее закрытие ЗА зоной — по какую оно сторону
                side = "bullish" if closes_after[outside[-1]] > top else "bearish"
    size = float(np.nanmean(atr(df).to_numpy())) or 1.0
    if family == "fvg":
        # три свечи FVG: зону обычно рисуют от 1-й или 2-й свечи — ищем рядом тройку, чья «дыра»
        # (high 1-й … low 3-й) лучше всего совпала с зоной
        highs, lows = df["high"].to_numpy(float), df["low"].to_numpy(float)
        near = range(max(0, start - 3), max(1, min(n - 2, start + 2)))
        def fit(i):
            gap = (highs[i], lows[i + 2]) if side == "bullish" else (highs[i + 2], lows[i])
            return abs(gap[0] - bottom) + abs(gap[1] - top)
        start = min(near, key=fit) if n >= 3 else start
        return {"kind": f"{side}_fvg", "start": start, "end": min(start + 2, n - 1), "zone": (bottom, top),
                "window": n - start, "significant": False, "filled": 0.0, "filled_at": None, "distance": 0.0,
                "from_zone": True}
    kind = f"{side}_{BLOCK[name]}"
    return {"kind": kind, "family": BLOCK[name], "start": start, "end": min(start + 1, n - 1),
            "swing": start, "zone": (bottom, top), "touched_at": None, "broken_at": None,
            "imbalance": False, "displacement": max(up, down) / size, "distance": 0.0, "from_zone": True}


LINE_BONUS = 6.0               # паттерн там, где ты провёл линию (уровень BOS, EQH, границы флага)


def _overlap(a0, a1, b0, b1):
    """Доля пересечения двух отрезков (по их объединению)."""
    a0, a1, b0, b1 = min(a0, a1), max(a0, a1), min(b0, b1), max(b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return max(0.0, min(a1, b1) - max(a0, b0)) / union if union > 0 else 0.0


def _line_match(item, lines, df):
    """Насколько находка совпадает с нарисованными линиями: 0..1."""
    if not lines:
        return 0.0
    size = float(np.nanmean(atr(df).to_numpy())) or 1.0
    kind = str(item.get("kind", ""))
    best = 0.0
    if kind in ("bos", "choch", "eqh", "eql"):
        first = item["touches"][0] if kind in ("eqh", "eql") else item["start"]
        for ln in lines:
            if not ln["flat"]:
                continue
            level = (ln["y0"] + ln["y1"]) / 2
            close = max(0.0, 1 - abs(level - item["level"]) / (0.6 * size))
            span = _overlap(first, item["end"], ln["x0"], ln["x1"])
            best = max(best, close * (0.4 + 0.6 * span))
    elif "upper" in item:                                       # флаг / вымпел — две линии тела
        sloped = [ln for ln in lines if ln["x1"] - ln["x0"] >= 3]
        for i, a in enumerate(sloped):
            for b in sloped[i + 1:]:
                top, bottom = (a, b) if a["y0"] + a["y1"] > b["y0"] + b["y1"] else (b, a)
                span = _overlap(item["upper"][0], item["upper"][2], top["x0"], top["x1"])
                height = max(item["upper"][1] - item["lower"][1], size)
                at = lambda ln, x: ln[1] + (ln[3] - ln[1]) * (x - ln[0]) / ((ln[2] - ln[0]) or 1)
                near = max(0.0, 1 - (abs(top["y1"] - at(item["upper"], top["x1"]))
                                     + abs(bottom["y1"] - at(item["lower"], bottom["x1"]))) / (2 * height))
                best = max(best, span * near)
    return best


def item_from_line(name, df, line):
    """BOS / CHoCH / EQH / EQL прямо по нарисованной горизонтальной линии — если правило на свечах
    со скрина такой паттерн не нашло (свеч мало, свинг не подтверждён и т.п.)."""
    family = FAMILY[name]
    n = len(df)
    level = (line["y0"] + line["y1"]) / 2
    x0 = int(max(0, min(n - 1, round(line["x0"]))))
    x1 = int(max(0, min(n - 1, round(line["x1"]))))
    closes, highs, lows = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    if family == "structure":
        # линия идёт от свинга: если уровень у хая свечи — это свинг-хай, слом — закрытие ВЫШЕ
        up = level >= (highs[x0] + lows[x0]) / 2
        broke = next((i for i in range(x0 + 1, n) if (closes[i] > level if up else closes[i] < level)), None)
        if broke is None:
            return None
        size = float(np.nanmean(atr(df).to_numpy())) or 1.0
        return {"kind": STRUCTURE[name], "direction": "up" if up else "down", "level": float(level),
                "start": x0, "end": int(broke), "window": int(broke - x0 + 1),
                "score": 0.9, "label": "", "from_line": True,
                "beyond": abs(closes[broke] - level) / size}
    if family == "eq":
        size = float(np.nanmean(atr(df).to_numpy())) or 1.0
        above = np.mean(closes[x0:x1 + 1] < level) >= 0.5          # цена под линией — это хаи
        values = highs if above else lows
        touches = [i for i in range(x0, x1 + 1) if abs(values[i] - level) <= 0.3 * size]
        if len(touches) < 2:
            touches = [x0, x1]
        swept = next((i for i in range(touches[-1] + 1, n)
                      if (highs[i] > level + 0.1 * size if above else lows[i] < level - 0.1 * size)), None)
        return {"kind": "eqh" if above else "eql", "level": float(level), "touches": touches,
                "start": touches[0], "end": touches[-1], "window": touches[-1] - touches[0] + 1,
                "swept_at": swept, "distance": 0.0, "score": 0.9, "label": "", "from_line": True}
    return None


def item_from_lines(name, df, lines):
    """Флаг / вымпел по двум нарисованным границам тела: флагшток — резкое движение
    до тела (до 20 свечей), направление — куда шёл флагшток."""
    pairs = [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]
             if min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) >= 2]
    if not pairs:
        return None
    a, b = max(pairs, key=lambda p: min(p[0]["x1"], p[1]["x1"]) - max(p[0]["x0"], p[1]["x0"]))
    top, bottom = (a, b) if a["y0"] + a["y1"] > b["y0"] + b["y1"] else (b, a)
    n = len(df)
    x0 = int(max(1, min(n - 1, round(max(top["x0"], bottom["x0"])))))
    x1 = int(max(x0, min(n - 1, round(min(top["x1"], bottom["x1"])))))
    highs, lows, closes = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    lo = max(0, x0 - 20)
    up_start = lo + int(np.argmin(lows[lo:x0 + 1]))
    down_start = lo + int(np.argmax(highs[lo:x0 + 1]))
    rise = highs[up_start:x0 + 1].max() - lows[up_start]
    fall = highs[down_start] - lows[down_start:x0 + 1].min()
    bullish = rise >= fall
    p0 = up_start if bullish else down_start
    p1 = p0 + int(np.argmax(highs[p0:x0 + 1]) if bullish else np.argmin(lows[p0:x0 + 1]))
    pole = (p0, float(lows[p0] if bullish else highs[p0]), p1, float(highs[p1] if bullish else lows[p1]))
    move = abs(pole[3] - pole[1])
    body = highs[p1:x1 + 1].max() - lows[p1:x1 + 1].min()
    shape = FLAG[name]
    kind = f"{'bullish' if bullish else 'bearish'}_{shape}"
    return {"kind": kind, "direction": "up" if bullish else "down", "pole": pole,
            "upper": (top["x0"], top["y0"], top["x1"], top["y1"]),
            "lower": (bottom["x0"], bottom["y0"], bottom["x1"], bottom["y1"]),
            "touches": 4, "violations": 0, "retrace": float(min(1.0, body / move)) if move > 0 else 0.5,
            "pole_move": (pole[3] - pole[1]) / pole[1] if pole[1] else 0.0, "score": 0.9,
            "start": int(p0), "end": int(x1), "window": int(x1 - p0 + 1), "label": "", "from_line": True}


SPEC_BONUS = 5.0               # паттерн подходит под твоё описание словами


def screenshot_fingerprint(name, df, drawn_levels=(), zones=(), lines=(), spec=None):
    """Твой паттерн на свечах со скрина: (отпечаток, находка) или (None, None).
    Если паттернов несколько — берём самый выраженный и ближе к правому краю;
    если ты нарисовал линию или закрасил зону — тот, что совпадает с твоей разметкой.
    Если правило на скрине такой паттерн не нашло — строим его по твоей разметке."""
    if name not in FAMILY or df is None or len(df) < 12:
        return None, None
    family = FAMILY[name]
    n = len(df)
    if family in ("block", "fvg"):                 # зона — закрашенный прямоугольник
        lines = [ln for ln in lines if not ln.get("zone")]
    else:                                          # уровни и флаги — линии; «зоны» из пары линий не нужны
        zones = [z for z in zones if family == "range"] if zones else []
    if len(zones) > 1:
        # зон несколько (пример + старая зона рядом, обрывок рамки) — главная та, что крупнее:
        # свою зону трейдер обычно тянет вправо до края графика
        zones = [max(zones, key=lambda z: (z.get("end", z["start"]) - z["start"] + 1) * (z["top"] - z["bottom"]))]
    best = None
    matched = 0.0
    candidates = all_items(name, df)
    if family == "structure" and lines:
        # BOS это или CHoCH, зависит от тренда ДО скрина, которого не видно; ты сам сказал, что это —
        # поэтому сравниваем с линией любые сломы и называем найденный так, как ты сказал
        candidates = [{**e, "kind": STRUCTURE[name]} for e in structure_events(df)]
    for item in candidates:
        feats = features_of(family, df, item)
        score = PROMINENCE[family](feats) + 0.5 * _end(item, n) / n
        prices = _price_range(item)
        if prices and drawn_levels and not lines:
            tolerance = 0.03 * (df["high"].max() - df["low"].min())
            if any(prices[0] - tolerance <= y <= prices[1] + tolerance for y in drawn_levels):
                score += DRAWN_BONUS
        match = max([_zone_match(item, z) for z in zones] + [_line_match(item, lines, df)])
        score += (ZONE_BONUS if zones else LINE_BONUS) * match
        if spec:                                   # из паттернов на скрине — тот, что ты описал словами
            import language
            fits, extra = language.check(spec, df, dict(item))
            score += SPEC_BONUS * fits + extra
        if best is None or score > best[0]:
            best = (score, item)
            matched = match
    if zones and matched < ZONE_MATCH:                 # правило не нашло то, что ты закрасил
        zone = max(zones, key=lambda z: z["start"])
        fallback = item_from_zone(name, df, zone)
        if fallback is not None:
            best = (0, fallback)
    flat = [ln for ln in lines if ln["flat"]]
    if not zones and flat and family in ("structure", "eq") and matched < ZONE_MATCH:
        fallback = item_from_line(name, df, max(flat, key=lambda ln: ln["x1"] - ln["x0"]))
        if fallback is not None:
            best = (0, fallback)
    sloped = [ln for ln in lines if not ln["flat"]]
    if len(sloped) < 2:                            # одна сторона вымпела / флага бывает почти горизонтальной
        sloped = list(lines)
    if not zones and family == "flags" and len(sloped) >= 2 and matched < 0.6:    # тело флага — по твоим линиям
        fallback = item_from_lines(name, df, sloped)
        if fallback is not None:
            best = (0, fallback)
    if best is None:
        return None, None
    item = {**best[1], "name": name}
    return make_fingerprint(name, df, item), item


# --- сравнение находки с примерами -------------------------------------------------------------
def visual_similarity(fp, df, item):
    """Вид участка с находкой против вида участка с примера. Участок рынка всегда один и тот же:
    до 40 свечей до паттерна и до 20 после (сколько есть) — тогда его «описание» считается
    один раз для всех примеров. У примера берём столько же свечей до и после."""
    window = fp.get("window")
    if not window:
        return None
    n = len(df)
    start, end = int(item["start"]), _end(item, n)
    before = min(CONTEXT_BEFORE, start, window["start"])
    after = min(CONTEXT_AFTER, n - 1 - end, len(window["ohlc"]) - 1 - window["end"])
    ref = np.asarray(window["ohlc"], dtype=float)[window["start"] - before:window["end"] + 1 + after]
    if len(ref) < 10:
        return None
    mirror = fp.get("bullish") != is_bullish(item)
    market = visual.normalize_ohlc(df, start - before, end + 1 + after, mirror=mirror)
    ref_low, ref_high = ref[:, 2].min(), ref[:, 1].max()
    if market is None or ref_high - ref_low <= 0:
        return None
    ref = (ref - ref_low) / (ref_high - ref_low)            # после обрезки — снова 0..1
    return visual.similarity(visual.look(ref), visual.look(market))


def similarity(fp, df, item):
    """Итоговая похожесть находки на пример 0..1."""
    feats = feature_similarity(fp["features"], features_of(fp["family"], df, item))
    look = visual_similarity(fp, df, item)
    if feats is None:
        return look
    if look is None:
        return feats
    return FEATURE_WEIGHT * feats + (1 - FEATURE_WEIGHT) * look


def best_match(fingerprints, df, item):
    """Самый похожий из примеров: (похожесть, номер примера) или (None, None)."""
    scored = [(s, i) for i, s in enumerate(similarity(fp, df, item) for fp in fingerprints) if s is not None]
    return max(scored) if scored else (None, None)


def describe(fp):
    """Отпечаток словами: «зона 1.2 ср. свечи, импульс 4.0 ср. свечи, объём ×2.1»."""
    parts = []
    for key, value in fp["features"].items():
        template = FEATURE_TEXT.get(key)
        if template is None:
            continue
        if FEATURE_KIND.get(key) == "flag":
            parts.append(template.format("да" if value else "нет"))
        elif key == "poc_rel":                                  # без процентов — словами
            parts.append("POC " + ("в нижней трети" if value < 0.34 else "в верхней трети" if value > 0.66
                                   else "в середине"))
        elif key == "retrace":
            parts.append("откат " + ("неглубокий" if value < 0.38 else "средний" if value < 0.62 else "глубокий"))
        else:
            parts.append(template.format(value))
    return ", ".join(parts)


def kind_text(item):
    """Что за паттерн — словами, без цен (на скрине цены в пикселях, они ничего не значат)."""
    kind = str(item.get("kind", ""))
    if kind in BLOCK_NAMES:
        return BLOCK_NAMES[kind] + (" с FVG" if item.get("imbalance") else "")
    if kind in FVG_NAMES:
        return FVG_NAMES[kind] + (" (значимый)" if item.get("significant") else "")
    if kind in ("bos", "choch"):
        return f"{kind.upper() if kind == 'bos' else 'CHoCH'} {'вверх' if item['direction'] == 'up' else 'вниз'}"
    if kind in EQ_NAMES:
        return EQ_NAMES[kind]
    if kind == "range":
        if item.get("drawn"):
            return "боковик по твоему прямоугольнику"
        return f"боковик, {item['end'] - item['start'] + 1} свечей"
    if kind in FLAG_NAMES:
        return FLAG_NAMES[kind]
    return item.get("name", "")


def range_from_box(df, box):
    """Боковик, который ТЫ нарисовал прямоугольником на скрине (vision.extract_range):
    границы — твои, касания считаем по свечам внутри, POC — твоя линия (или по объёму).
    Длину в свечах не берём: на мелком скрине свечи сливаются и их число неточное."""
    start, end = box["start"], box["end"]
    bottom, top = box["bottom"], box["top"]
    height = top - bottom
    if height <= 0 or end - start < 10:
        return None, None
    highs = df["high"].to_numpy(dtype=float)[start:end + 1]
    lows = df["low"].to_numpy(dtype=float)[start:end + 1]
    window = max(2, (end - start) // 20)
    tops = [i for i in pivot_highs(list(highs), window) if highs[i] >= top - EDGE * height]
    bottoms = [i for i in pivot_lows(list(lows), window) if lows[i] <= bottom + EDGE * height]
    poc = box.get("poc")
    if poc is None and "volume" in df:
        profile, centers = volume_profile(highs, lows, df["volume"].to_numpy(dtype=float)[start:end + 1],
                                          bottom, top)
        poc = float(centers[int(np.argmax(profile))]) if profile.sum() > 0 else None
    item = {"kind": "range", "name": "Range", "box": (bottom, top), "poc": poc, "start": start, "end": end,
            "window": end - start + 1, "touches": len(tops) + len(bottoms), "score": 1.0, "drawn": True,
            "touch_points": [(start + i, float(highs[i])) for i in tops] + [(start + i, float(lows[i])) for i in bottoms]}
    fp = make_fingerprint("Range", df, item)
    fp["features"].pop("length", None)
    return fp, item
