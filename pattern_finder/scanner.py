"""Поиск паттернов пользователя по рынку и отправка результатов."""

from catalog import LINE_PATTERNS, RULE_PATTERNS, SHAPE_PATTERNS
from chart import plot_findings
from market import candles_to_df, get_candles
from matcher import find_best_match_multi
from notifier import send_photo, tradingview_link
from shape import photo_to_shape
from smc import find_fvg
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, find_line_pattern

INTERVAL = "4h"
WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = 3              # None — искать по всей истории (для проверки)
MIN_SCORE = 0.85
CANDLES = 300
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


# --- детекторы: получили свечи, вернули находку или None ------------------------------------
def find_shape(df, shape):
    """Паттерн «по форме» с картинки пользователя."""
    start, score, window_len = find_best_match_multi(df["close"].tolist(), shape, WINDOWS, RECENT)
    if score < MIN_SCORE:
        return None
    return {"score": score, "start": start, "window": window_len}


def find_lines(df, pattern_name, preferred_kind):
    """Треугольник / клин / канал по линиям тренда, формирующийся прямо сейчас."""
    found = find_line_pattern(df, FAMILY_BY_PATTERN[pattern_name], preferred_kind)
    if found is None or found["score"] < MIN_SCORE:
        return None
    found["label"] = KIND_NAMES[found["kind"]]
    return found


def find_rule(df, pattern_name):
    """SMC-паттерны по правилам на свечах."""
    detect = {"FVG": find_fvg}[pattern_name]
    found = detect(df)
    if found is None or found["score"] < MIN_SCORE:
        return None
    return found


def build_detectors(patterns):
    """Готовит детектор под каждый паттерн пользователя. Возвращает (детекторы, пропущенные).
    Одинаковые паттерны (два «Треугольника» одного типа, два FVG) проверяются один раз."""
    detectors = []
    skipped = []
    seen = set()
    for p in patterns:
        name = p["name"]
        if name in SHAPE_PATTERNS:
            try:
                shape = photo_to_shape(p["photo"])
            except (ValueError, OSError):
                skipped.append(name)
                continue
            detectors.append((name, lambda df, s=shape: find_shape(df, s)))
            continue

        key = (name, p.get("kind"))
        if key in seen:
            continue
        seen.add(key)
        if name in LINE_PATTERNS:
            kind = p.get("kind")
            detectors.append((name, lambda df, n=name, k=kind: find_lines(df, n, k)))
        elif name in RULE_PATTERNS:
            detectors.append((name, lambda df, n=name: find_rule(df, n)))
        elif name not in skipped:
            skipped.append(name)
    return detectors, skipped


def scan_user(patterns):
    """Все паттерны пользователя × все монеты. Свечи каждой монеты качаем один раз."""
    detectors, skipped = build_detectors(patterns)
    findings = {}
    if not detectors:
        return findings, skipped

    for symbol in SYMBOLS:
        candles = get_candles(symbol, INTERVAL, CANDLES)
        if not candles:
            continue
        df = candles_to_df(candles)
        items = []
        for name, detect in detectors:
            found = detect(df)
            if found is not None:
                found["name"] = name
                items.append(found)
        if items:
            findings[symbol] = {"df": df, "items": items}
    return findings, skipped


def describe_item(item):
    text = f"{item['name']} ({item['score']:.0%})"
    if item.get("label"):
        text += f" — {item['label']}"
    return text


def send_scan_results(findings, chat_id, limit=5):
    ordered = sorted(findings.items(), key=lambda kv: len(kv[1]["items"]), reverse=True)
    for symbol, data in ordered[:limit]:
        df = data["df"]
        items = data["items"]
        best = max(items, key=lambda it: it["score"])
        path = f"scan_{chat_id}_{symbol}.png"
        plot_findings(df, symbol, items, path)

        names = "\n".join(f"• {describe_item(it)}" for it in items)
        title = "🔥 Комбинация" if len(items) > 1 else "🔔 Паттерн"
        begin = df.loc[best["start"], "open_time"]
        end = df.loc[best["start"] + best["window"] - 1, "open_time"]
        caption = (
            f"{title} на {symbol} {INTERVAL}\n"
            f"{names}\n"
            f"Период: {begin:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC\n"
            f"{tradingview_link(symbol, INTERVAL)}"
        )
        send_photo(path, caption, chat_id)
