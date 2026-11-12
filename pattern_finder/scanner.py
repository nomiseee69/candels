"""Поиск паттернов пользователя по рынку: алерты «сейчас» и история.

Результат любого поиска — «карточки»: монета + таймфрейм + найденные паттерны.
Бот рисует по карточке график, подпись и разметку для TradingView.
"""

from catalog import FLAG_PATTERNS, FLAG_SHAPE, LINE_PATTERNS, RULE_PATTERNS, SHAPE_PATTERNS
from chart import plot_coin, plot_findings, plot_history
from blocks import find_block
from history import (
    HORIZON, coin_report, history_blocks, non_overlapping, history_equal, history_fvg, history_lines,
    history_shapes, history_structure, plural, summarize,
)
from market import FALLBACK_SYMBOLS, candles_to_df, fetch_many, get_candles
from matcher import find_best_match
from shapes_check import shape_label, validate
from notifier import tradingview_link
from shape import photo_to_shape
from similarity import SIMILAR_ENOUGH, best_match, reference_from_screenshot
from fingerprint import best_match as best_fingerprint
from confluence import caption_block, confluences, score as confluence_score, trend
from smc import find_fvg
from flags import find_flag, history_flags
from ranges import find_range, history_ranges
from storage import DEFAULT_SETTINGS
import language
import quality
import strength
from texts import finding_name
from structure import find_break, find_equal
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, WINDOWS as LINE_WINDOWS, find_line_pattern

ALL_INTERVALS = ["1h", "4h", "12h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24, "1w": 168}
DEFAULT_INTERVALS = ["4h"]

WINDOWS = [20, 30, 40, 60, 80, 120]
RECENT = 3              # алерты: паттерн должен закончиться на последних 3 свечах
MIN_SCORE = 0.85
CANDLES = 1000          # глубина графика: 1H ≈ 41 день, 4H ≈ 5.5 месяца, 1D ≈ 2.7 года
HISTORY_CANDLES = 3000  # история, разметка и примеры: 1H ≈ 4 месяца, 4H ≈ 1.4 года, 1D ≈ 8 лет (3 запроса)
MAX_CARDS = 8           # графиками присылаем 8 лучших находок, остальные — списком
OVERVIEW_LINES = 40     # сколько строк в списке остальных находок
HISTORY_CHARTS = 3      # графики истории — для 3 монет с наибольшим числом случаев
REPORT_COINS = 15       # в отчёте истории подробно — 15 монет, остальные одной строкой
COIN_OPTIONS = [10, 30, 50, 100]   # сколько самых торгуемых монет сканировать (выбор в меню)
STRUCTURE_ON_CHART = 8             # на графике истории — последние 8 сломов структуры

STRUCTURE_KIND = {"BOS": "bos", "CHoCH": "choch"}
BLOCK_FAMILY = {"Order Block": "ob", "Breaker Block": "breaker", "Mitigation Block": "mitigation"}


def _significant(settings):
    """Только «значимые» FVG по грубому правилу — лишь когда оценка силы выключена.
    Иначе эксперт (strength.py) сам оценивает КАЖДЫЙ FVG: импульс, слом, объём, ликвидность,
    дисконт, тренд — и хорошие FVG, не прошедшие грубое правило, не теряются."""
    if quality.strictness(settings) != "off":
        return False
    return (settings or DEFAULT_SETTINGS).get("fvg", "significant") == "significant"


def rate(df, found):
    """Оценка силы находки экспертом (пишет strength / level в находку). Отбор — потом,
    в scan_user: там же учитываются твои 👍/👎."""
    if found is not None:
        strength.evaluate(df, found)
    return found


def period_text(interval, candles):
    days = candles * INTERVAL_HOURS[interval] // 24
    if days >= 730:
        return f"~{plural(days // 365, 'год', 'года', 'лет')}"
    return f"~{plural(days, 'день', 'дня', 'дней')}"


# --- детекторы «сейчас»: получили свечи, вернули находку или None ------------------------------
def spec_bonus(specs):
    """Описания словами → добавка к выбору кандидата: подходящий под описание главнее (+10),
    плюс по 0.5 за каждое выполненное «желательно». Без описаний — None."""
    if not specs:
        return None

    def bonus(df, item):
        fits, extra = language.check_any(specs, df, dict(item))
        return (10 if fits else 0) + 0.5 * extra
    return bonus


AFTER_BREAKOUT = 10        # «пробой подтверждён» в описании: фигура, пробитая не дальше 10 свечей назад


def wants_breakout(specs):
    return any(k in (s.get("require") or []) for s in specs or [] for k in ("confirmed", "retest", "body_close"))


def find_shape(df, shapes, builtin=None, name=None, extra=None):
    """Паттерн «по форме»: сравниваем с КАЖДЫМ примером пользователя и берём лучший.
    shapes — формы со всех его скринов (у одного паттерна их может быть несколько);
    builtin — для каждой формы: это встроенный эталон, а не скрин пользователя?
    name — название: двойное дно / ГиП дополнительно проверяются по точкам (shapes_check)."""
    closes = df["close"].tolist()
    best = None
    for number, shape in enumerate(shapes, start=1):
        for window_len in WINDOWS:
            if len(closes) < window_len:
                continue
            start, score = find_best_match(closes, shape, window_len, RECENT)
            if score < MIN_SCORE or (best is not None and extra is None and score <= best["score"]):
                continue
            checked = validate(name, df, start, window_len) if name else {}
            if checked is None:
                continue                                    # похоже на «W», но по точкам — не двойное дно
            found = {"score": float(score), "start": start, "window": window_len, "example": number,
                     **({"name": name} if checked else {}), **checked}
            found["_rank"] = score + (extra(df, {**found, "name": name}) if extra else 0)   # описание словами
            if best is not None and found["_rank"] <= best["_rank"]:
                continue
            best = found
            if checked:
                best["label"] = shape_label(name, checked)
    if best is None:
        return None
    best.pop("_rank")
    if builtin and builtin[best["example"] - 1]:
        best["builtin"] = True                              # «похоже на эталон», а не «на твой пример»
    if len(shapes) == 1:
        best.pop("example")
    return best


def find_lines(df, pattern_name, preferred_kind, references=(), similar_only=False, specs=()):
    """Треугольник / клин / канал по линиям тренда, формирующийся прямо сейчас.
    references — эталоны с рисунков пользователя: считаем «похоже на твой пример»;
    similar_only — не присылать фигуры, похожие меньше чем на 60%;
    specs — описания словами: из фигур-кандидатов (разные окна) берём ту, что под них подходит."""
    def compare(d, figure):
        score, index = best_match(d, figure, references)
        if index is not None and len(references) > 1:
            figure["example"] = index + 1                   # «похоже на пример №2»
        return score
    compare = compare if references else None
    def described(d, figure):
        fits, extra = language.check_any(specs, d, figure)
        return (10 if fits else 0) + 0.5 * extra
    found = find_line_pattern(df, FAMILY_BY_PATTERN[pattern_name], preferred_kind, similarity=compare,
                              extra=described if specs else None)
    if wants_breakout(specs) and (found is None or not language.check_any(specs, df, dict(found))[0]):
        # в описании «пробой подтверждён / ретест» — берём фигуру, которую пробили недавно
        n = len(df)
        recent = [f for f in history_lines(df, FAMILY_BY_PATTERN[pattern_name], LINE_WINDOWS, MIN_SCORE)
                  if max(f["upper"][2], f["lower"][2]) >= n - 1 - AFTER_BREAKOUT and f.get("breakout")]
        for f in recent:
            if compare:
                f["similarity"] = compare(df, f)
        recent = [f for f in recent if language.check_any(specs, df, dict(f))[0]]
        if recent:
            found = max(recent, key=lambda f: f["score"] + 0.3 * (f.get("similarity") or 0))
            found["window"] = n - found["start"]
    if found is None or found["score"] < MIN_SCORE:
        return None
    if similar_only and references and (found.get("similarity") or 0) < SIMILAR_ENOUGH:
        return None
    found["label"] = KIND_NAMES[found["kind"]]
    return found


def find_rule(df, pattern_name, settings=None, fingerprints=(), specs=()):
    """SMC-паттерны по правилам на свечах. fingerprints — отпечатки примеров пользователя:
    из подходящих по правилу находок берётся самая похожая на его пример,
    и в находку пишется «похоже на твой пример N%».
    specs — описания словами (language.py): из кандидатов берём тот, что под них подходит."""
    similar = None
    if fingerprints:
        similar = lambda item: best_fingerprint(fingerprints, df, item)[0] or 0
    # из кандидатов — самый сильный по оценке эксперта (а с примерами — похожий на пример И сильный)
    choose = similar
    if quality.strictness(settings) != "off" or specs:
        choose = strength.chooser(df, similar)
    if specs:                                        # твоё описание — главнее всего остального
        base = choose

        def choose(item, base=base):
            fits, extra = language.check_any(specs, df, item)
            return (10 if fits else 0) + 0.5 * extra + base(item)
    if pattern_name == "FVG":
        found = find_fvg(df, significant_only=_significant(settings), choose=choose)
    elif pattern_name in STRUCTURE_KIND:
        found = find_break(df, STRUCTURE_KIND[pattern_name], choose=choose)
    elif pattern_name in BLOCK_FAMILY:
        strict = (settings or DEFAULT_SETTINGS).get("ob_volume", True)     # «только эталонные блоки»
        found = find_block(df, BLOCK_FAMILY[pattern_name], volume_only=strict, choose=choose, strict=strict)
    elif pattern_name == "Range":
        found = find_range(df, extra=spec_bonus(specs))
    else:                                            # "EQH / EQL"
        found = find_equal(df, choose=choose)
    if found is None or found["score"] < MIN_SCORE:
        return None
    found = rate(df, found)
    return compare_with_examples(found, df, fingerprints, settings)


def compare_with_examples(found, df, fingerprints, settings=None):
    """«Похоже на твой пример N%» для SMC, Range и флагов. С настройкой «только похожие» —
    непохожие (< 60%) не присылаем."""
    if found is None or not fingerprints:
        return found
    score, index = best_fingerprint(fingerprints, df, found)
    if score is None:
        return found
    found["similarity"] = score
    if len(fingerprints) > 1:
        found["example"] = index + 1
    if (settings or {}).get("similar_only") and score < SIMILAR_ENOUGH:
        return None
    return found


def _specs(patterns, name):
    """Описания словами примеров паттерна (language.parse). Пример без описания — {} (без условий).
    Если описаний нет совсем — пустой список."""
    examples = [p for p in patterns if p["name"] == name and not p.get("auto")]   # 👍-примеры описаний не имеют
    if not any(p.get("spec") for p in examples):
        return []
    return [p.get("spec") or {} for p in examples]


def _fingerprints(patterns, name):
    """Отпечатки всех примеров паттерна (со скринов и сохранённых с рынка)."""
    return [p["fingerprint"] for p in patterns if p["name"] == name and p.get("fingerprint")]


def _min_score(found):
    return found if found is not None and found["score"] >= MIN_SCORE else None


def build_detectors(patterns, settings=None):
    """Детектор под каждый паттерн пользователя. Возвращает (детекторы, пропущенные).
    Одинаковые паттерны (два «Треугольника» одного типа, два FVG) проверяются один раз."""
    detectors = []
    skipped = []
    seen = set()
    for p in patterns:
        name = p["name"]
        if name in SHAPE_PATTERNS:
            if name in seen:
                continue
            seen.add(name)
            # все примеры паттерна — один детектор: нет дублей «Двойное дно» в одном алерте
            shapes, builtin = [], []
            for q in patterns:
                if q["name"] != name:
                    continue
                try:
                    shapes.append(q["shape"] if q.get("shape") else photo_to_shape(q.get("photo")))
                    builtin.append(bool(q.get("builtin")))
                except (ValueError, OSError, TypeError):
                    pass
            if not shapes:
                skipped.append(name)
                continue
            bonus = spec_bonus(_specs(patterns, name))
            detectors.append((name, lambda df, s=shapes, b=builtin, n=name, x=bonus:
                              rate(df, find_shape(df, s, b, n, x))))
            continue

        key = (name, p.get("kind"))
        if key in seen:
            continue
        seen.add(key)
        if name in LINE_PATTERNS:
            kind = p.get("kind")
            # все рисунки этого паттерна и типа — эталоны «похоже на твой пример»
            references = [ref for ref in (q.get("reference") or (q.get("photo") and reference_from_screenshot(q["photo"]))
                                          for q in patterns
                                          if q["name"] == name and q.get("kind") == kind) if ref]
            similar_only = bool((settings or {}).get("similar_only"))
            specs = _specs(patterns, name)
            detectors.append((name, lambda df, n=name, k=kind, r=references, s=specs:
                              rate(df, find_lines(df, n, k, r, similar_only, s))))
        elif name in FLAG_PATTERNS:
            fps = _fingerprints(patterns, name)
            bonus = spec_bonus(_specs(patterns, name))
            detectors.append((name, lambda df, s=FLAG_SHAPE[name], f=fps, x=bonus:
                              compare_with_examples(rate(df, _min_score(find_flag(df, s, x))), df, f, settings)))
        elif name in RULE_PATTERNS:
            fps = _fingerprints(patterns, name)
            specs = _specs(patterns, name)
            detectors.append((name, lambda df, n=name, f=fps, s=specs: find_rule(df, n, settings, f, s)))
        elif name not in skipped:
            skipped.append(name)
    return detectors, skipped


# --- карточки -----------------------------------------------------------------------------------------
def describe_item(item):
    """Строка находки в алерте: только паттерн и направление («Order Block ↑»)."""
    return finding_name(item)


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
def scan_user(patterns, intervals=DEFAULT_INTERVALS, fetch=None, symbols=None, settings=None,
              judge=None):
    """Все паттерны пользователя × монеты × таймфреймы. Свечи качаем один раз на монету и ТФ,
    сразу для многих монет параллельно.
    fetch — функция загрузки свечей (автоскан передаёт свою, с кэшем на всех пользователей).
    settings — настройки пользователя: какие FVG искать, только ли комбинации.
    judge — «чему бот научился» по оценкам 👍/👎 (feedback.make_judge): отсеивает
    то, что пользователю не нравится, и добавляет «подходит тебе N%».
    Возвращает (ВСЕ карточки — комбинации и лучшие первыми, пропущенные паттерны)."""
    fetch = fetch or get_candles
    symbols = symbols or FALLBACK_SYMBOLS
    settings = settings or DEFAULT_SETTINGS
    min_items = 2 if settings.get("combos_only") else 1
    detectors, skipped = build_detectors(patterns, settings)
    cards = []
    if not detectors:
        return cards, skipped
    charts = {}                                       # (монета, ТФ) → график: для трендов всех ТФ

    for interval in intervals:
        all_candles = fetch_many(symbols, interval, CANDLES, fetch)
        for symbol in symbols:
            candles = all_candles[symbol]
            if not candles or len(candles) < max(WINDOWS):
                continue                              # молодая монета — мало свечей для поиска
            df = candles_to_df(candles)
            charts[(symbol, interval)] = df
            items = []
            for name, detect in detectors:
                specs = _specs(patterns, name)
                allowed = language.timeframes(specs)
                if allowed and interval not in allowed:
                    continue                          # в описании — только другие таймфреймы
                found = detect(df)
                if found is None:
                    continue
                found["name"] = name
                if judge is not None and getattr(judge, "disliked", None) and "start" in found and \
                        f"{symbol}|{interval}|{name}|{int(df['open_time'].iloc[found['start']].timestamp())}" \
                        in judge.disliked:
                    continue                          # 👎 — эту находку ты уже отклонил
                fits, extra = language.check_any(specs, df, found, interval)
                if not fits:
                    continue                          # не такой, как ты описал словами
                if specs:
                    found["described"] = True
                if judge is not None:
                    keep, fit = judge(name, df, found)      # твои 👍/👎: веса и строгость под тебя
                    if not keep:
                        continue
                    if fit is not None:
                        found["fit"] = fit
                if not strength.passes(found, settings):   # слабые по оценке силы не присылаем
                    continue
                items.append(found)
            items = declutter_figures(items)            # треугольник и канал на одном месте — одна фигура
            if len(items) >= min_items:
                cards.append({
                    "symbol": symbol, "interval": interval, "df": df, "items": items,
                    "history": False, "caption": alert_caption(symbol, interval, df, items),
                })
    # тренды ВСЕХ просканированных ТФ монеты — для «совпадения по таймфреймам»,
    # даже на тех ТФ, где паттернов не нашлось (считаем только для монет с находками)
    trends = {}
    for card in cards:
        symbol = card["symbol"]
        if symbol not in trends:
            trends[symbol] = {iv: trend(charts[(symbol, iv)]) for iv in intervals if (symbol, iv) in charts}
        card["trends"] = trends[symbol]
    cards.sort(key=lambda c: (len(c["items"]), max(it.get("strength") or it["score"] for it in c["items"])),
               reverse=True)
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
                finding_name(it) for it in f["items"])
            for f in coin["frames"])
        lines.append(f"• {coin['symbol']} — {frames}")
    if len(coins) > limit:
        lines.append(f"…и ещё {len(coins) - limit}")
    return "\n".join(lines)


# --- одна монета — одно сообщение со всеми таймфреймами ----------------------------------------------
def _coin_rank(coin):
    """Порядок монет: похожесть на твой пример > совпадение по ТФ > сила находки > комбинация на одном ТФ >
    находки на нескольких ТФ > больше находок > оценка."""
    frames = coin["frames"]
    # главное — насколько находки похожи на ТВОИ примеры (шагами по 10%), потом совпадения ТФ
    likeness = [it.get("similarity") for f in frames for it in f["items"] if it.get("similarity") is not None]
    power = [it.get("strength") or 0 for f in frames for it in f["items"]]
    return (round(max(likeness), 1) if likeness else 0,
            confluence_score(coin.get("confluence", [])),
            round(max(power, default=0), 1),                  # сила лучшей находки по оценке эксперта
            max(len(f["items"]) for f in frames), len(frames),
            sum(len(f["items"]) for f in frames),
            max(it["score"] for f in frames for it in f["items"]))


def coin_caption(coin):
    """Подпись монеты: находки по таймфреймам, от младшего к старшему."""
    frames = coin["frames"]
    symbol = coin["symbol"]
    lowest = frames[0]["interval"]
    if coin["history"]:
        return "\n\n".join(f["caption"] for f in frames)
    block = caption_block(coin.get("confluence", []))
    if len(frames) == 1:
        caption = frames[0]["caption"]
        if not block:
            return caption
        head, link = caption.rsplit("\n", 1)              # совпадения — перед ссылкой
        return f"{head}\n\n{block}\n{link}"
    tfs = ", ".join(f["interval"].upper() for f in frames)
    combo = any(len(f["items"]) > 1 for f in frames)
    linked = confluence_score(coin.get("confluence", [])) > 0
    icon = "🔗" if linked else "🔥" if combo else "🔔"
    lines = [f"{icon} {symbol}: находки на {len(frames)} таймфреймах ({tfs})"]
    for f in frames:
        mark = "  🔥 комбинация" if len(f["items"]) > 1 else ""
        lines.append(f"\n⏱ {f['interval'].upper()}{mark}")
        lines += [f"• {describe_item(it)}" for it in f["items"]]
    if block:
        lines.append(f"\n{block}")
    lines.append(f"\n{tradingview_link(symbol, lowest)}")
    return "\n".join(lines)


def _span(frame, item):
    df = frame["df"]
    end = min(item["start"] + item.get("window", 1) - 1, len(df) - 1)
    return df["open_time"].iloc[int(item["start"])], df["open_time"].iloc[end]


def prune_figures(coin):
    """Фигуры монеты на всех ТФ: сначала каналы, потом флаги / клинья / формы, треугольники — в конце;
    при равенстве — со старшего ТФ. Не больше двух, и не на одном и том же участке времени
    (тот же канал на 1H внутри канала 4H — лишний)."""
    figures = [(f, it) for f in coin["frames"] for it in f["items"] if is_figure(it)]
    figures.sort(key=lambda e: (FIGURE_PRIORITY[figure_family(e[1])], INTERVAL_HOURS[e[0]["interval"]],
                                e[1]["score"]), reverse=True)
    kept = []
    for frame, item in figures:
        start, end = _span(frame, item)
        overlaps = any(start <= k_end and end >= k_start for _, _, k_start, k_end in kept)
        if len(kept) < MAX_COIN_FIGURES and not overlaps:
            kept.append((frame, item, start, end))
    keep_ids = {id(item) for _, item, _, _ in kept}
    for frame in coin["frames"]:
        before = len(frame["items"])
        frame["items"] = [it for it in frame["items"] if not is_figure(it) or id(it) in keep_ids]
        if len(frame["items"]) != before and frame["items"]:
            frame["caption"] = alert_caption(frame["symbol"], frame["interval"], frame["df"], frame["items"])
    coin["frames"] = [f for f in coin["frames"] if f["items"]]


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
        if not coin["history"]:
            prune_figures(coin)
    result = [coin for coin in result if coin["frames"]]
    for coin in result:
        coin["confluence"] = [] if coin["history"] else confluences(coin)
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
        if name in seen:
            continue
        seen.add(name)
        if name in SHAPE_PATTERNS:
            shapes = []
            for q in patterns:                          # все примеры паттерна — одним заданием
                if q["name"] == name:
                    try:
                        shapes.append(q["shape"] if q.get("shape") else photo_to_shape(q.get("photo")))
                    except (ValueError, OSError, TypeError):
                        pass
            if not shapes:
                skipped.append(name)
                continue
            jobs.append((name, "shape", lambda df, ss=shapes, n=name: non_overlapping(
                [f for s in ss for f in history_shapes(df, s, WINDOWS, MIN_SCORE, n)])))
        elif name in LINE_PATTERNS:
            family = FAMILY_BY_PATTERN[name]
            jobs.append((name, "lines", lambda df, f=family: history_lines(df, f, LINE_WINDOWS, MIN_SCORE)))
        elif name == "FVG":
            jobs.append((name, "fvg", lambda df: history_fvg(df, significant)))
        elif name in STRUCTURE_KIND:
            jobs.append((name, "structure", lambda df, k=STRUCTURE_KIND[name]: history_structure(df, k)))
        elif name == "EQH / EQL":
            jobs.append((name, "eq", history_equal))
        elif name in BLOCK_FAMILY:
            strict = (settings or DEFAULT_SETTINGS).get("ob_volume", True)
            jobs.append((name, "block", lambda df, f=BLOCK_FAMILY[name], s=strict: history_blocks(df, f, s)))
        elif name == "Range":
            jobs.append((name, "range", history_ranges))
        elif name in FLAG_PATTERNS:
            jobs.append((name, "flags", lambda df, s=FLAG_SHAPE[name]: history_flags(df, s)))
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
            if family_kind == "block" and not (f["broken_at"] is None and f["touched_at"] is None
                                               and f["distance"] <= 0.05):
                continue
            items.append({**f, "name": name})
    return items


def _coin_weight(entries):
    """Порядок монет в отчёте: сначала по фигурам (формы, линии), потом по SMC."""
    main = sum(len(found) for _, kind, found in entries if kind in ("shape", "lines", "flags", "range"))
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
                "caption": (f"{symbol} {interval}: все случаи за период "
                            "(что было после каждого — в отчёте «📊 История» выше)\n"
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


FIGURE_PRIORITY = {"channel": 3, "flag": 2, "pennant": 2, "wedge": 2, "shape": 2, "triangle": 1}
MAX_COIN_FIGURES = 2          # фигур на монету (все ТФ вместе) — не больше двух, остальное шум


def figure_family(item):
    kind = str(item.get("kind", ""))
    if kind.startswith("channel"):
        return "channel"
    if kind.endswith("_flag"):
        return "flag"
    if kind.endswith("_pennant"):
        return "pennant"
    if kind.endswith("wedge"):
        return "wedge"
    if item.get("points"):
        return "shape"
    return "triangle"


def is_figure(item):
    return "upper" in item or bool(item.get("points"))


def declutter_figures(items):
    """Треугольник, клин и канал на одном и том же участке — это одна фигура, увиденная
    по-разному. Оставляем канал (он понятнее), потом клин, потом треугольник;
    при равенстве — где больше касаний."""
    figures = sorted((it for it in items if "upper" in it),
                     key=lambda it: (FIGURE_PRIORITY[figure_family(it)], it["score"]), reverse=True)
    kept = []
    for fig in figures:
        if all(_overlap_share(fig, k) < 0.5 for k in kept):
            kept.append(fig)
    return [it for it in items if "upper" not in it] + sorted(kept, key=lambda it: it["start"])


def _past_items(jobs, df, current, settings=None, specs=None, interval=None):
    """Все прошлые случаи паттернов пользователя на графике df, кроме тех,
    что совпадают с текущими находками того же паттерна (иначе линии двоятся).
    Живые зоны и ликвидность оценивает эксперт: слабые на график не попадают."""
    from pine import is_alive
    current_ranges = [(it["name"], *_time_range(df_cur, it)) for df_cur, it in current]
    past = []
    for name, _, run in jobs:
        for f in run(df):
            start, end = _time_range(df, f)
            duplicate = any(
                name == cur_name and start <= cur_end and end >= cur_start
                for cur_name, cur_start, cur_end in current_ranges
            )
            if duplicate:
                continue
            item = {**f, "name": name}
            if quality.family_of(item) in ("fvg", "block", "eq") and is_alive(item):
                strength.evaluate(df, item)                   # насколько сильна зона, если цена придёт сейчас
                if not strength.passes(item, settings):
                    continue
                if not language.check_any((specs or {}).get(name, []), df, item, interval)[0]:
                    continue                                  # не такая, как ты описал словами
            past.append(item)
    return declutter_figures(past)


def selected_markup(patterns, coin, chosen, settings=None):
    """Разметка только выбранных находок (кнопка «🧩 Паттерны»).
    chosen — [(таймфрейм, находка)]. Прошлые случаи берём только для тех же паттернов
    и таймфреймов — на графике не будет ничего лишнего."""
    names_by_tf = {}
    for tf, item in chosen:
        names_by_tf.setdefault(tf, set()).add(item["name"])
    names = set().union(*names_by_tf.values()) if names_by_tf else set()
    frames = [f for f in coin.get("frames", [coin]) if f["interval"] in names_by_tf]
    groups = full_markup([p for p in patterns if p["name"] in names], {**coin, "frames": frames},
                         list(names_by_tf), settings)
    chosen_ids = {id(item) for _, item in chosen}
    result = []
    for df, items, history, tf in groups:
        if tf not in names_by_tf:
            continue
        if history:
            items = [it for it in items if it["name"] in names_by_tf[tf]]
        else:
            items = [it for it in items if id(it) in chosen_ids]
        result.append((df, items, history, tf))
    return result


def full_markup(patterns, coin, intervals=None, settings=None):
    """Группы для pine_script по ВСЕМ таймфреймам монеты: прошлые случаи паттернов
    пользователя (HISTORY_CANDLES свечей каждого ТФ) + текущие находки.
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
            specs = {p["name"]: _specs(patterns, p["name"]) for p in patterns}
            groups.append((df, _past_items(jobs, df, [(frame["df"], it) for it in current], settings,
                                           specs, interval),
                           True, interval))
        if current:
            groups.append((frame["df"], current, False, interval))
    return groups
