"""Поиск паттернов пользователя по рынку: алерты «сейчас» и история.

Результат любого поиска — «карточки»: монета + таймфрейм + найденные паттерны.
Бот рисует по карточке график, подпись и разметку для TradingView.
"""

from catalog import LINE_PATTERNS, RULE_PATTERNS, SHAPE_PATTERNS
from chart import plot_findings, plot_history
from history import (
    HORIZON, coin_report, history_fvg, history_lines, history_shapes, plural, summarize,
)
from market import candles_to_df, get_candles
from matcher import find_best_match_multi
from notifier import tradingview_link
from shape import photo_to_shape
from smc import find_fvg
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, WINDOWS as LINE_WINDOWS, find_line_pattern

ALL_INTERVALS = ["1h", "4h", "12h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24, "1w": 168}
DEFAULT_INTERVALS = ["4h"]

WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = 3              # алерты: паттерн должен закончиться на последних 3 свечах
MIN_SCORE = 0.85
CANDLES = 300           # для поиска «сейчас»
HISTORY_CANDLES = 500   # для истории (Binance отдаёт до 1000)
MAX_CARDS = 8           # не больше 8 карточек за одно сканирование
HISTORY_CHARTS = 3      # графики истории — для 3 монет с наибольшим числом случаев
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


def period_text(interval, candles):
    days = candles * INTERVAL_HOURS[interval] // 24
    if days >= 730:
        return f"~{plural(days // 365, 'год', 'года', 'лет')}"
    return f"~{plural(days, 'день', 'дня', 'дней')}"


# --- детекторы «сейчас»: получили свечи, вернули находку или None ------------------------------
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
    """Детектор под каждый паттерн пользователя. Возвращает (детекторы, пропущенные).
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


# --- карточки -----------------------------------------------------------------------------------------
def describe_item(item):
    text = f"{item['name']} ({item['score']:.0%})"
    if item.get("label"):
        text += f" — {item['label']}"
    return text


def alert_caption(symbol, interval, df, items):
    best = max(items, key=lambda it: it["score"])
    names = "\n".join(f"• {describe_item(it)}" for it in items)
    title = "🔥 Комбинация" if len(items) > 1 else "🔔 Паттерн"
    begin = df["open_time"].iloc[best["start"]]
    end = df["open_time"].iloc[min(best["start"] + best["window"] - 1, len(df) - 1)]
    return (
        f"{title} на {symbol} {interval}\n"
        f"{names}\n"
        f"Период: {begin:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC\n"
        f"{tradingview_link(symbol, interval)}"
    )


def draw_card(card, path):
    """Рисует график карточки в файл path."""
    if card["history"]:
        plot_history(card["df"], card["symbol"], card["items"], path)
    else:
        plot_findings(card["df"], card["symbol"], card["items"], path)
    return path


# --- сканирование «сейчас» -----------------------------------------------------------------------------
def scan_user(patterns, intervals=DEFAULT_INTERVALS, fetch=None):
    """Все паттерны пользователя × монеты × таймфреймы. Свечи качаем один раз на монету и ТФ.
    fetch — функция загрузки свечей (автоскан передаёт свою, с кэшем на всех пользователей).
    Возвращает (карточки — комбинации первыми, пропущенные паттерны)."""
    fetch = fetch or get_candles
    detectors, skipped = build_detectors(patterns)
    cards = []
    if not detectors:
        return cards, skipped

    for interval in intervals:
        for symbol in SYMBOLS:
            candles = fetch(symbol, interval, CANDLES)
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
                cards.append({
                    "symbol": symbol, "interval": interval, "df": df, "items": items,
                    "history": False, "caption": alert_caption(symbol, interval, df, items),
                })
    cards.sort(key=lambda c: (len(c["items"]), max(it["score"] for it in c["items"])), reverse=True)
    return cards[:MAX_CARDS], skipped


# --- история ---------------------------------------------------------------------------------------------
def build_history_jobs(patterns):
    """Что и как искать в истории: [(название, тип, функция df → находки)], пропущенные."""
    jobs = []
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
            jobs.append((name, "shape", lambda df, s=shape: history_shapes(df, s, WINDOWS, MIN_SCORE)))
            continue
        if name in seen:
            continue
        seen.add(name)
        if name in LINE_PATTERNS:
            family = FAMILY_BY_PATTERN[name]
            jobs.append((name, "lines", lambda df, f=family: history_lines(df, f, LINE_WINDOWS, MIN_SCORE)))
        elif name == "FVG":
            jobs.append((name, "fvg", history_fvg))
        else:
            skipped.append(name)
    return jobs, skipped


def _chart_items(entries):
    """Что рисовать на графике истории: все случаи форм и фигур + открытые FVG рядом с ценой."""
    items = []
    for name, family_kind, found in entries:
        for f in found:
            if family_kind == "fvg":
                if f["filled_at"] is None and f["distance"] <= 0.05:
                    items.append({**f, "name": name})
            else:
                items.append({**f, "name": name})
    return items


def history_user(patterns, intervals=DEFAULT_INTERVALS):
    """История по всем монетам. Возвращает (тексты отчётов, карточки с графиками, пропущенные).
    Отчёт — по каждой монете: что нашлось, когда и чем закончилось."""
    jobs, skipped = build_history_jobs(patterns)
    reports = []
    cards = []
    if not jobs:
        return reports, cards, skipped

    for interval in intervals:
        per_job = {i: [] for i in range(len(jobs))}
        coins = []
        for symbol in SYMBOLS:
            candles = get_candles(symbol, interval, HISTORY_CANDLES)
            if not candles:
                continue
            df = candles_to_df(candles)
            entries = []
            for i, (name, family_kind, run) in enumerate(jobs):
                found = run(df)
                for f in found:
                    f["_n"] = len(df)
                per_job[i].extend(found)
                entries.append((name, family_kind, found))
            weight = sum(len(found) for _, kind, found in entries if kind != "fvg")
            if any(found for _, _, found in entries):
                coins.append((weight, symbol, df, entries))

        coins.sort(key=lambda c: c[0], reverse=True)
        header = f"📊 История {interval} за {period_text(interval, HISTORY_CANDLES)} — что нашлось на каждой монете"
        blocks = [coin_report(symbol, df, entries) for _, symbol, df, entries in coins]
        totals = ["Итого по всем монетам:"] + [
            summarize(name, kind, per_job[i]) for i, (name, kind, _) in enumerate(jobs)
        ]
        note = (f"Стрелка — куда пошла цена за {HORIZON} свечей после паттерна. "
                "Это статистика, а не торговый сигнал.")
        reports.append("\n\n".join([header] + (blocks or ["Ничего не нашлось."]) + ["\n".join(totals), note]))

        for weight, symbol, df, entries in coins[:HISTORY_CHARTS]:
            items = _chart_items(entries)
            if weight == 0 or not items:
                continue
            cards.append({
                "symbol": symbol, "interval": interval, "df": df, "items": items, "history": True,
                "caption": (f"{symbol} {interval}: все случаи за период\n"
                            "зелёный — цена потом выросла, красный — упала\n"
                            f"{tradingview_link(symbol, interval)}"),
            })
    return reports, cards, skipped


# --- полная разметка для TradingView: всё, что было на графике, + текущая находка ------------------
def _time_range(df, item):
    start = df["open_time"].iloc[item["start"]]
    end = df["open_time"].iloc[min(item["start"] + item["window"] - 1, len(df) - 1)]
    return start, end


def full_markup(patterns, card):
    """Группы для pine_script: все прошлые случаи паттернов пользователя на этой монете
    и таймфрейме (500 свечей) + текущие находки карточки. Если текущая находка совпадает
    с прошлым случаем того же паттерна, прошлый не рисуем — чтобы не было двойных линий."""
    jobs, _ = build_history_jobs(patterns)
    groups = []
    candles = get_candles(card["symbol"], card["interval"], HISTORY_CANDLES)
    current = [] if card["history"] else card["items"]
    current_ranges = [(it["name"], *_time_range(card["df"], it)) for it in current]

    if candles:
        df = candles_to_df(candles)
        past = []
        for name, _, run in jobs:
            for f in run(df):
                start, end = _time_range(df, f)
                duplicate = any(
                    name == cur_name and start <= cur_end and end >= cur_start
                    for cur_name, cur_start, cur_end in current_ranges
                )
                if not duplicate:
                    past.append({**f, "name": name})
        groups.append((df, past, True))
    if current:
        groups.append((card["df"], current, False))
    return groups
