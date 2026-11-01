"""Поиск паттернов пользователя по рынку: алерты «сейчас» и история.

Результат любого поиска — «карточки»: монета + таймфрейм + найденные паттерны.
Бот рисует по карточке график, подпись и разметку для TradingView.
"""

from catalog import LINE_PATTERNS, RULE_PATTERNS, SHAPE_PATTERNS
from chart import plot_coin, plot_findings, plot_history
from blocks import find_block
from history import (
    HORIZON, coin_report, history_blocks, history_equal, history_fvg, history_lines,
    history_shapes, history_structure, plural, summarize,
)
from market import FALLBACK_SYMBOLS, candles_to_df, fetch_many, get_candles
from matcher import find_best_match_multi
from notifier import tradingview_link
from shape import photo_to_shape
from smc import find_fvg
from storage import DEFAULT_SETTINGS
from texts import quality_text
from structure import find_break, find_equal
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, WINDOWS as LINE_WINDOWS, find_line_pattern

ALL_INTERVALS = ["1h", "4h", "12h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24, "1w": 168}
DEFAULT_INTERVALS = ["4h"]

WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = 3              # алерты: паттерн должен закончиться на последних 3 свечах
MIN_SCORE = 0.85
CANDLES = 1000          # глубина графика: 1H ≈ 41 день, 4H ≈ 5.5 месяца, 1D ≈ 2.7 года
HISTORY_CANDLES = 1000  # та же глубина для истории и разметки (один запрос на монету и ТФ)
MAX_CARDS = 8           # графиками присылаем 8 лучших находок, остальные — списком
OVERVIEW_LINES = 40     # сколько строк в списке остальных находок
HISTORY_CHARTS = 3      # графики истории — для 3 монет с наибольшим числом случаев
REPORT_COINS = 15       # в отчёте истории подробно — 15 монет, остальные одной строкой
COIN_OPTIONS = [10, 30, 50, 100]   # сколько самых торгуемых монет сканировать (выбор в меню)
STRUCTURE_ON_CHART = 8             # на графике истории — последние 8 сломов структуры

STRUCTURE_KIND = {"BOS": "bos", "CHoCH": "choch"}
BLOCK_FAMILY = {"Order Block": "ob", "Breaker Block": "breaker", "Mitigation Block": "mitigation"}


def _significant(settings):
    return (settings or DEFAULT_SETTINGS).get("fvg", "significant") == "significant"


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


def find_rule(df, pattern_name, settings=None):
    """SMC-паттерны по правилам на свечах."""
    if pattern_name == "FVG":
        found = find_fvg(df, significant_only=_significant(settings))
    elif pattern_name in STRUCTURE_KIND:
        found = find_break(df, STRUCTURE_KIND[pattern_name])
    elif pattern_name in BLOCK_FAMILY:
        found = find_block(df, BLOCK_FAMILY[pattern_name])
    else:                                            # "EQH / EQL"
        found = find_equal(df)
    if found is None or found["score"] < MIN_SCORE:
        return None
    return found


def build_detectors(patterns, settings=None):
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
            detectors.append((name, lambda df, n=name: find_rule(df, n, settings)))
        elif name not in skipped:
            skipped.append(name)
    return detectors, skipped


# --- карточки -----------------------------------------------------------------------------------------
def describe_item(item):
    quality = quality_text(item)
    text = f"{item['name']} ({quality})" if quality else item["name"]
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
def scan_user(patterns, intervals=DEFAULT_INTERVALS, fetch=None, symbols=None, settings=None):
    """Все паттерны пользователя × монеты × таймфреймы. Свечи качаем один раз на монету и ТФ,
    сразу для многих монет параллельно.
    fetch — функция загрузки свечей (автоскан передаёт свою, с кэшем на всех пользователей).
    settings — настройки пользователя: какие FVG искать, только ли комбинации.
    Возвращает (ВСЕ карточки — комбинации и лучшие первыми, пропущенные паттерны)."""
    fetch = fetch or get_candles
    symbols = symbols or FALLBACK_SYMBOLS
    settings = settings or DEFAULT_SETTINGS
    min_items = 2 if settings.get("combos_only") else 1
    detectors, skipped = build_detectors(patterns, settings)
    cards = []
    if not detectors:
        return cards, skipped

    for interval in intervals:
        all_candles = fetch_many(symbols, interval, CANDLES, fetch)
        for symbol in symbols:
            candles = all_candles[symbol]
            if not candles or len(candles) < max(WINDOWS):
                continue                              # молодая монета — мало свечей для поиска
            df = candles_to_df(candles)
            items = []
            for name, detect in detectors:
                found = detect(df)
                if found is not None:
                    found["name"] = name
                    items.append(found)
            items = declutter_figures(items)            # треугольник и канал на одном месте — одна фигура
            if len(items) >= min_items:
                cards.append({
                    "symbol": symbol, "interval": interval, "df": df, "items": items,
                    "history": False, "caption": alert_caption(symbol, interval, df, items),
                })
    cards.sort(key=lambda c: (len(c["items"]), max(it["score"] for it in c["items"])), reverse=True)
    return cards, skipped


def cards_overview(coins, limit=OVERVIEW_LINES):
    """Короткий список монет, которые не поместились графиками:
    • SUIUSDT — 1H: FVG · 4H: Треугольник (6 касаний)"""
    if not coins:
        return ""
    lines = [f"📋 Ещё {plural(len(coins), 'монета', 'монеты', 'монет')} с находками "
             "— нажми на монету внизу, чтобы получить график и разметку:"]
    for coin in coins[:limit]:
        frames = " · ".join(
            f"{f['interval'].upper()}: " + ", ".join(
                f"{it['name']} ({quality_text(it)})" if quality_text(it) else it["name"] for it in f["items"])
            for f in coin["frames"])
        lines.append(f"• {coin['symbol']} — {frames}")
    if len(coins) > limit:
        lines.append(f"…и ещё {len(coins) - limit}")
    return "\n".join(lines)


# --- одна монета — одно сообщение со всеми таймфреймами ----------------------------------------------
def _coin_rank(coin):
    """Порядок монет: комбинация на одном ТФ > находки на нескольких ТФ > больше находок > оценка."""
    frames = coin["frames"]
    return (max(len(f["items"]) for f in frames), len(frames),
            sum(len(f["items"]) for f in frames),
            max(it["score"] for f in frames for it in f["items"]))


def coin_caption(coin):
    """Подпись монеты: находки по таймфреймам, от младшего к старшему."""
    frames = coin["frames"]
    symbol = coin["symbol"]
    lowest = frames[0]["interval"]
    if coin["history"]:
        return "\n\n".join(f["caption"] for f in frames)
    if len(frames) == 1:
        return frames[0]["caption"]
    tfs = ", ".join(f["interval"].upper() for f in frames)
    combo = any(len(f["items"]) > 1 for f in frames)
    lines = [f"{'🔥' if combo else '🔔'} {symbol}: находки на {len(frames)} таймфреймах ({tfs})"]
    for f in frames:
        mark = "  🔥 комбинация" if len(f["items"]) > 1 else ""
        lines.append(f"\n⏱ {f['interval'].upper()}{mark}")
        lines += [f"• {describe_item(it)}" for it in f["items"]]
    lines.append(f"\n{tradingview_link(symbol, lowest)}")
    return "\n".join(lines)


def group_by_coin(cards):
    """Карточки «монета + таймфрейм» → по одной на монету со всеми её таймфреймами.
    Возвращает монеты: {symbol, frames (по возрастанию ТФ), history, caption}."""
    coins = {}
    for card in cards:
        key = (card["symbol"], card["history"])
        coins.setdefault(key, {"symbol": card["symbol"], "history": card["history"], "frames": []})
        coins[key]["frames"].append(card)
    result = list(coins.values())
    for coin in result:
        coin["frames"].sort(key=lambda f: INTERVAL_HOURS[f["interval"]])
        coin["caption"] = coin_caption(coin)
    result.sort(key=_coin_rank, reverse=True)
    return result


def draw_coin(coin, path):
    """Один таймфрейм — обычный график; несколько — панели друг под другом."""
    if len(coin["frames"]) == 1:
        return draw_card(coin["frames"][0], path)
    plot_coin(coin["symbol"], [(f["interval"], f["df"], f["items"], f["history"]) for f in coin["frames"]], path)
    return path


# --- история ---------------------------------------------------------------------------------------------
def build_history_jobs(patterns, settings=None):
    """Что и как искать в истории: [(название, тип, функция df → находки)], пропущенные."""
    significant = _significant(settings)
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
            jobs.append((name, "fvg", lambda df: history_fvg(df, significant)))
        elif name in STRUCTURE_KIND:
            jobs.append((name, "structure", lambda df, k=STRUCTURE_KIND[name]: history_structure(df, k)))
        elif name == "EQH / EQL":
            jobs.append((name, "eq", history_equal))
        elif name in BLOCK_FAMILY:
            jobs.append((name, "block", lambda df, f=BLOCK_FAMILY[name]: history_blocks(df, f)))
        else:
            skipped.append(name)
    return jobs, skipped


def _chart_items(entries):
    """Что рисовать на графике истории: все случаи форм и фигур, открытые FVG рядом с ценой,
    неснятая ликвидность и последние сломы структуры."""
    items = []
    for name, family_kind, found in entries:
        if family_kind == "structure":
            found = found[-STRUCTURE_ON_CHART:]
        for f in found:
            if family_kind == "fvg" and not (f["filled_at"] is None and f["distance"] <= 0.05):
                continue
            if family_kind == "eq" and f["swept_at"] is not None:
                continue
            if family_kind == "block" and not (f["broken_at"] is None and f["distance"] <= 0.05):
                continue
            items.append({**f, "name": name})
    return items


def _coin_weight(entries):
    """Порядок монет в отчёте: сначала по фигурам (формы, линии), потом по SMC."""
    main = sum(len(found) for _, kind, found in entries if kind in ("shape", "lines"))
    smc = sum(len(found) for _, kind, found in entries if kind in ("structure", "eq", "block"))
    return main, smc


def history_user(patterns, intervals=DEFAULT_INTERVALS, symbols=None, fetch=None, settings=None):
    """История по всем монетам. Возвращает (тексты отчётов, карточки с графиками, пропущенные).
    Отчёт — по каждой монете: что нашлось, когда и чем закончилось."""
    symbols = symbols or FALLBACK_SYMBOLS
    fetch = fetch or get_candles
    jobs, skipped = build_history_jobs(patterns, settings)
    reports = []
    cards = []
    if not jobs:
        return reports, cards, skipped

    for interval in intervals:
        per_job = {i: [] for i in range(len(jobs))}
        coins = []
        all_candles = fetch_many(symbols, interval, HISTORY_CANDLES, fetch)
        for symbol in symbols:
            candles = all_candles[symbol]
            if not candles or len(candles) < max(WINDOWS):
                continue
            df = candles_to_df(candles)
            entries = []
            for i, (name, family_kind, run) in enumerate(jobs):
                found = run(df)
                for f in found:
                    f["_n"] = len(df)
                per_job[i].extend(found)
                entries.append((name, family_kind, found))
            weight = _coin_weight(entries)
            if any(found for _, _, found in entries):
                coins.append((weight, symbol, df, entries))

        coins.sort(key=lambda c: c[0], reverse=True)
        header = (f"📊 История {interval} за {period_text(interval, HISTORY_CANDLES)} — "
                  f"что нашлось ({plural(len(symbols), 'монета', 'монеты', 'монет')} в поиске)")
        blocks = [coin_report(symbol, df, entries) for _, symbol, df, entries in coins[:REPORT_COINS]]
        rest = coins[REPORT_COINS:]
        if rest:
            blocks.append(f"Ещё находки на {plural(len(rest), 'монете', 'монетах', 'монетах')}: "
                          + ", ".join(symbol for _, symbol, _, _ in rest))
        totals = ["Итого по всем монетам:"] + [
            summarize(name, kind, per_job[i]) for i, (name, kind, _) in enumerate(jobs)
        ]
        note = (f"Стрелка — куда пошла цена за {HORIZON} свечей после паттерна. "
                "Это статистика, а не торговый сигнал.")
        reports.append("\n\n".join([header] + (blocks or ["Ничего не нашлось."]) + ["\n".join(totals), note]))

        for weight, symbol, df, entries in coins[:HISTORY_CHARTS]:
            items = _chart_items(entries)
            if not items:
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


def _overlap_share(a, b):
    """Какая доля более короткого из двух участков (start, window) перекрыта другим."""
    a0, a1 = a["start"], a["start"] + a["window"]
    b0, b1 = b["start"], b["start"] + b["window"]
    shorter = max(min(a1 - a0, b1 - b0), 1)
    return max(0, min(a1, b1) - max(a0, b0)) / shorter


def declutter_figures(items):
    """Треугольник, клин и канал на одном и том же участке — это одна фигура, увиденная
    по-разному. Оставляем ту, у которой больше касаний, — без «каши» из линий."""
    figures = sorted((it for it in items if "upper" in it), key=lambda it: it["score"], reverse=True)
    kept = []
    for fig in figures:
        if all(_overlap_share(fig, k) < 0.5 for k in kept):
            kept.append(fig)
    return [it for it in items if "upper" not in it] + sorted(kept, key=lambda it: it["start"])


def _past_items(jobs, df, current):
    """Все прошлые случаи паттернов пользователя на графике df, кроме тех,
    что совпадают с текущими находками того же паттерна (иначе линии двоятся)."""
    current_ranges = [(it["name"], *_time_range(df_cur, it)) for df_cur, it in current]
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
    return declutter_figures(past)


def full_markup(patterns, coin, intervals=None, settings=None):
    """Группы для pine_script по ВСЕМ таймфреймам монеты: прошлые случаи паттернов
    пользователя (1000 свечей каждого ТФ) + текущие находки.
    coin — монета из group_by_coin (или одна карточка — для старых вызовов).
    intervals — таймфреймы пользователя: разметка строится и для тех, где сейчас ничего нет.
    Группа = (df, находки, это_история, таймфрейм)."""
    frames = coin.get("frames", [coin])
    by_interval = {f["interval"]: f for f in frames}
    wanted = list(dict.fromkeys(list(intervals or []) + list(by_interval)))
    wanted.sort(key=lambda i: INTERVAL_HOURS[i])
    jobs, _ = build_history_jobs(patterns, settings)
    groups = []
    for interval in wanted:
        frame = by_interval.get(interval)
        current = [] if frame is None or frame["history"] else frame["items"]
        candles = get_candles(coin["symbol"], interval, HISTORY_CANDLES)
        if candles:
            df = candles_to_df(candles)
            groups.append((df, _past_items(jobs, df, [(frame["df"], it) for it in current]), True, interval))
        if current:
            groups.append((frame["df"], current, False, interval))
    return groups
