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
На рынке каждая находка получает такой же отпечаток и «похоже на твой пример N%»:
65% — числа, 35% — вид. Из нескольких находок на монете выбирается самая похожая.
"""

import numpy as np

from blocks import all_blocks, volume_ratios
from flags import find_flag, history_flags
from ranges import find_range, history_ranges
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


def screenshot_fingerprint(name, df, drawn_levels=()):
    """Твой паттерн на свечах со скрина: (отпечаток, находка) или (None, None).
    Если паттернов несколько — берём самый выраженный и ближе к правому краю;
    если ты нарисовал горизонтальную линию — тот, что у этой линии."""
    if name not in FAMILY or df is None or len(df) < 20:
        return None, None
    family = FAMILY[name]
    n = len(df)
    best = None
    for item in all_items(name, df):
        feats = features_of(family, df, item)
        score = PROMINENCE[family](feats) + 0.5 * _end(item, n) / n
        prices = _price_range(item)
        if prices and drawn_levels:
            tolerance = 0.03 * (df["high"].max() - df["low"].min())
            if any(prices[0] - tolerance <= y <= prices[1] + tolerance for y in drawn_levels):
                score += DRAWN_BONUS
        if best is None or score > best[0]:
            best = (score, item)
    if best is None:
        return None, None
    item = {**best[1], "name": name}
    return make_fingerprint(name, df, item), item


# --- сравнение находки с примерами -------------------------------------------------------------
def visual_similarity(fp, df, item):
    """Вид участка с находкой против вида участка с примера (выравниваем по началу паттерна)."""
    window = fp.get("window")
    if not window:
        return None
    ref = np.asarray(window["ohlc"], dtype=float)
    offset = window["start"]
    lo = int(item["start"]) - offset
    cut_left = max(0, -lo)
    lo = max(0, lo)
    hi = min(len(df), lo + len(ref) - cut_left)
    ref = ref[cut_left:cut_left + (hi - lo)]
    if hi - lo < 10:
        return None
    mirror = fp.get("bullish") != is_bullish(item)
    market = visual.normalize_ohlc(df, lo, hi, mirror=mirror)
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
        return f"боковик, {item['end'] - item['start'] + 1} свечей"
    if kind in FLAG_NAMES:
        return FLAG_NAMES[kind]
    return item.get("name", "")
