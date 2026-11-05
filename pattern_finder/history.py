"""История: все случаи паттерна на графике и что было после каждого.

Отвечает на вопрос «стоит ли доверять этому паттерну»: сколько раз он
встречался и куда потом пошла цена.
"""

import numpy as np

from texts import where_text

from shape import normalize, resample
from blocks import all_blocks
from smc import find_all_fvg
from structure import equal_levels, structure_events
from texts import plural
from trendlines import KIND_NAMES, _analyze_window, line_value, shift

HORIZON = 10               # исход смотрим через 10 свечей после паттерна
SHAPE_STEP = 2             # сдвиг окна при поиске форм (каждые 2 свечи)
LINE_STEP = 5              # сдвиг окна при поиске фигур из линий
FVG_FILL_WINDOW = 20       # «FVG отработал» = закрылся за 20 свечей


# --- общее ----------------------------------------------------------------------------------
def outcome(df, end, horizon=HORIZON):
    """Что сделала цена за horizon свечей после свечи end. None — паттерн слишком свежий."""
    if end + horizon >= len(df):
        return None
    base = float(df["close"].iloc[end])
    after = df.iloc[end + 1:end + 1 + horizon]
    return {
        "change": float(after["close"].iloc[-1]) / base - 1,
        "max_up": float(after["high"].max()) / base - 1,
        "max_down": float(after["low"].min()) / base - 1,
    }


def non_overlapping(found):
    """Из пересекающихся находок оставляет лучшие: одно двойное дно не должно
    засчитываться 20 раз со сдвигом окна на пару свечей."""
    kept = []
    # при равной оценке берём случай, который заканчивается позже: он ближе к пробою
    for f in sorted(found, key=lambda f: (f["score"], f["start"] + f["window"]), reverse=True):
        a0, a1 = f["start"], f["start"] + f["window"]
        if all(a1 <= k["start"] or a0 >= k["start"] + k["window"] for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f["start"])


# --- формы с картинки ---------------------------------------------------------------------------
def history_shapes(df, shape, windows, min_score, name=None):
    """Все случаи формы за период. name — двойное дно / ГиП проверяются по точкам (shapes_check)."""
    from shapes_check import shape_label, validate
    prices = df["close"].to_numpy()
    found = []
    for w in windows:
        for start in range(0, len(prices) - w + 1, SHAPE_STEP):
            segment = normalize(resample(prices[start:start + w], len(shape)))
            score = float(np.corrcoef(shape, segment)[0, 1])
            if score >= min_score:
                found.append({"start": start, "window": w, "score": score})
    if name:                                  # сначала проверка по точкам, потом убираем пересечения
        from structure import atr
        sizes = atr(df).to_numpy()
        checked = []
        for f in found:
            extra = validate(name, df, f["start"], f["window"], sizes)
            if extra is None:
                continue
            f.update(extra)
            if extra:
                f["label"] = shape_label(name, extra)
            checked.append(f)
        found = checked
    found = non_overlapping(found)
    for f in found:
        f["outcome"] = outcome(df, f["start"] + f["window"] - 1)
    return found


# --- треугольники, клинья, каналы ------------------------------------------------------------------
def _breakout(df, found, horizon=HORIZON):
    """Куда цена вышла из фигуры за horizon свечей: "up", "down" или None."""
    x1 = found["upper"][2]
    closes = df["close"].to_numpy()
    for x in range(int(x1) + 1, min(int(x1) + 1 + horizon, len(closes))):
        if closes[x] > line_value(found["upper"], x):
            return "up"
        if closes[x] < line_value(found["lower"], x):
            return "down"
    return None


def history_lines(df, family, windows, min_score):
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    n = len(df)
    found = []
    for w in windows:
        # длинные окна двигаем крупнее: фигура на 250 свечей не меняется от сдвига на 5
        for end in range(w, n + 1, max(LINE_STEP, w // 10)):
            f = _analyze_window(highs[end - w:end], lows[end - w:end], closes[end - w:end])
            if f is None or f["kind"] not in family or f["score"] < min_score:
                continue
            found.append(shift(f, end - w))
    found = non_overlapping(found)
    for f in found:
        end = f["start"] + f["window"] - 1
        f["outcome"] = outcome(df, end)
        f["breakout"] = _breakout(df, f) if end + 1 < n else None
    return found


# --- FVG ----------------------------------------------------------------------------------------------
def history_fvg(df, significant_only=False):
    gaps = find_all_fvg(df, lookback=len(df), include_filled=True, significant_only=significant_only)
    for g in gaps:
        g["fill_candles"] = g["filled_at"] - g["end"] if g["filled_at"] is not None else None
    return sorted(gaps, key=lambda g: g["start"])


# --- BOS / CHoCH и равные хаи/лои ----------------------------------------------------------------
def history_structure(df, kind):
    """Все сломы структуры типа kind ("bos" / "choch") и что было через HORIZON свечей."""
    events = [e for e in structure_events(df) if e["kind"] == kind]
    for e in events:
        e["outcome"] = outcome(df, e["end"])
    return events


def history_equal(df):
    """Все уровни равных хаёв/лоёв: сняли ли ликвидность и через сколько свечей."""
    levels = equal_levels(df)
    for lv in levels:
        lv["sweep_candles"] = lv["swept_at"] - lv["end"] if lv["swept_at"] is not None else None
    return levels


def history_blocks(df, family, strict=True):
    """Все блоки семейства ("ob" / "breaker" / "mitigation") и что было после первого
    возвращения цены в зону: оттолкнулась ли она в сторону блока.
    strict — только эталонные (сняли ликвидность, поглощение, объём — см. blocks.py)."""
    blocks = [blk for blk in all_blocks(df) if blk["family"] == family and (not strict or blk.get("textbook", True))]
    for blk in blocks:
        touched = blk["touched_at"]
        blk["outcome"] = outcome(df, touched) if touched is not None else None
    return blocks


def _held(blk):
    """После ретеста цена ушла в сторону блока (бычий — вверх, медвежий — вниз)?"""
    return (blk["outcome"]["change"] > 0) == blk["kind"].startswith("bullish")


def _followed(e):
    """Цена через HORIZON свечей ушла в сторону слома?"""
    return (e["outcome"]["change"] > 0) == (e["direction"] == "up")


# --- сводка по нескольким монетам -----------------------------------------------------------------
def summarize(name, family_kind, results):
    """results: список находок со всех монет. Возвращает строку для Telegram."""
    if not results:
        return f"• {name}: не встречался"

    if family_kind == "fvg":
        total = len(results)
        bull = sum(r["kind"] == "bullish_fvg" for r in results)
        judged = [r for r in results if r["end"] + FVG_FILL_WINDOW < r["_n"]]
        quick = [r for r in judged if r["fill_candles"] is not None and r["fill_candles"] <= FVG_FILL_WINDOW]
        open_now = sum(r["filled_at"] is None for r in results)
        line = f"• {name}: {total} шт. (бычьих {bull}, медвежьих {total - bull}), сейчас открыто {open_now}"
        if judged:
            line += f"\n   закрылись за {FVG_FILL_WINDOW} свечей: {len(quick)} из {len(judged)}"
        return line

    if family_kind == "structure":
        ups = sum(r["direction"] == "up" for r in results)
        line = (f"• {name}: {plural(len(results), 'случай', 'случая', 'случаев')} "
                f"(вверх {ups}, вниз {len(results) - ups})")
        judged = [r for r in results if r.get("outcome")]
        if judged:
            followed = sum(_followed(r) for r in judged)
            line += f"\n   через {HORIZON} свечей цена ушла в сторону слома в {followed} из {len(judged)}"
        return line

    if family_kind == "eq":
        swept = [r for r in results if r["swept_at"] is not None]
        line = (f"• {name}: {plural(len(results), 'уровень', 'уровня', 'уровней')}, "
                f"ликвидность снята у {len(swept)}, ещё не снята у {len(results) - len(swept)}")
        if swept:
            line += f"\n   в среднем сняли через {np.mean([r['sweep_candles'] for r in swept]):.0f} свечей"
        return line

    if family_kind == "block":
        bull = sum(r["kind"].startswith("bullish") for r in results)
        broken = sum(r["broken_at"] is not None for r in results)
        line = (f"• {name}: {len(results)} шт. (бычьих {bull}, медвежьих {len(results) - bull}), "
                f"пробито {broken}, действует {len(results) - broken}")
        judged = [r for r in results if r.get("outcome")]
        if judged:
            held = sum(_held(r) for r in judged)
            line += (f"\n   после ретеста цена ушла в сторону блока в {held} из {len(judged)} "
                     f"за {HORIZON} свечей")
        return line

    line = f"• {name}: {plural(len(results), 'случай', 'случая', 'случаев')}"
    if family_kind in ("flags", "range"):
        ups = sum(r.get("breakout") == "up" for r in results)
        downs = sum(r.get("breakout") == "down" for r in results)
        line += f"\n   выход за {HORIZON} свечей: вверх {ups}, вниз {downs}, ещё внутри {len(results) - ups - downs}"
        if family_kind == "flags":
            along = sum(r.get("breakout") == r.get("direction") for r in results)
            line += f" (по флагштоку — {along})"
    if family_kind == "lines":
        kinds = {}
        for r in results:
            kinds[KIND_NAMES[r["kind"]]] = kinds.get(KIND_NAMES[r["kind"]], 0) + 1
        line += " (" + ", ".join(f"{k} {v}" for k, v in kinds.items()) + ")"
        ups = sum(r.get("breakout") == "up" for r in results)
        downs = sum(r.get("breakout") == "down" for r in results)
        line += f"\n   пробой за {HORIZON} свечей: вверх {ups}, вниз {downs}, без пробоя {len(results) - ups - downs}"

    judged = [r["outcome"] for r in results if r.get("outcome")]
    if judged:
        rose = sum(o["change"] > 0 for o in judged)
        line += f"\n   через {HORIZON} свечей цена выше в {rose} из {len(judged)}"
    return line


# --- отчёт по одной монете: что и когда нашлось ---------------------------------------------------
SHORT_KIND = {
    "symmetric": "симметричный", "ascending": "восходящий", "descending": "нисходящий",
    "rising_wedge": "восходящий", "falling_wedge": "нисходящий",
    "channel_up": "восходящий", "channel_down": "нисходящий", "channel_flat": "боковой",
}
MAX_DATES = 4


def _date(df, i):
    i = max(0, min(int(i), len(df) - 1))
    return f"{df['open_time'].iloc[i]:%d.%m}"


def _result(o):
    if o is None:
        return "рано судить"
    return "↑" if o["change"] > 0 else "↓"


def describe_occurrence(df, family_kind, f):
    end = f["start"] + f["window"] - 1
    text = _date(df, end)
    if family_kind == "structure":
        return f"{text} {'вверх' if f['direction'] == 'up' else 'вниз'} ({_result(f.get('outcome'))})"
    if family_kind in ("flags", "range"):
        if family_kind == "flags":
            text += " бычий" if f.get("direction") == "up" else " медвежий"
        if f.get("breakout"):
            text += ", выход " + ("вверх" if f["breakout"] == "up" else "вниз")
        return f"{text} ({_result(f.get('outcome'))})"
    if family_kind == "lines":
        text += f" {SHORT_KIND[f['kind']]}"
        if f.get("touches"):
            text += f" ({plural(f['touches'], 'касание', 'касания', 'касаний')})"
        if f.get("breakout"):
            text += ", пробой " + ("вверх" if f["breakout"] == "up" else "вниз")
            result = f.get("outcome")
            # вышли из фигуры в одну сторону, а через 10 свечей цена оказалась с другой
            if result and (result["change"] > 0) != (f["breakout"] == "up"):
                text += " — ложный"
    return f"{text} ({_result(f.get('outcome'))})"


def coin_report(symbol, df, entries):
    """entries: [(название, тип, находки)]. Текст про одну монету, свежие случаи первыми."""
    lines = [f"🪙 {symbol}"]
    for name, family_kind, found in entries:
        if not found:
            continue
        if family_kind == "fvg":
            open_gaps = [g for g in found if g["filled_at"] is None]
            line = f"• FVG: {len(found)} за период, открыто {len(open_gaps)}"
            if open_gaps:
                g = min(open_gaps, key=lambda g: g["distance"])
                bottom, top = g["zone"]
                where = where_text(g["distance"], "цена внутри")
                kind = "бычий" if g["kind"] == "bullish_fvg" else "медвежий"
                line += f"\n   ближайший открытый: {kind} {bottom:.6g}–{top:.6g}, {where}"
            lines.append(line)
            continue
        if family_kind == "block":
            active = [blk for blk in found if blk["broken_at"] is None and blk["touched_at"] is None]
            line = f"• {name}: {len(found)} за период, не отработано {len(active)}"
            if active:
                line += "\n   ближайший: " + min(active, key=lambda blk: blk["distance"])["label"]
            lines.append(line)
            continue
        if family_kind == "eq":
            waiting = [lv for lv in found if lv["swept_at"] is None]
            line = f"• {name}: {plural(len(found), 'уровень', 'уровня', 'уровней')}, не снято {len(waiting)}"
            if waiting:
                line += "\n   ближайший: " + min(waiting, key=lambda lv: lv["distance"])["label"]
            lines.append(line)
            continue
        # свежие первыми: у сломов структуры — по свече слома, у фигур — по началу
        newest = sorted(found, key=lambda f: f["end"] if family_kind == "structure" else f["start"],
                        reverse=True)
        shown = ", ".join(describe_occurrence(df, family_kind, f) for f in newest[:MAX_DATES])
        more = len(found) - MAX_DATES
        lines.append(f"• {name} ×{len(found)}: {shown}" + (f" и ещё {more}" if more > 0 else ""))
    return "\n".join(lines)


def read_lines_figure(name, path):
    """Треугольник / клин / канал на СКРИНШОТЕ без нарисованных линий: читаем свечи и ищем
    фигуру тем же правилом, что и на рынке. Возвращает (свечи, фигура) или None."""
    from trendlines import FAMILY_BY_PATTERN, WINDOWS
    from vision import extract_candles
    df = extract_candles(path)
    if df is None or len(df) < 30:
        return None
    n = len(df)
    windows = [w for w in WINDOWS if w <= n] + [w for w in (40, 65, 100) if w <= n]   # как на рынке + промежуточные
    found = history_lines(df, FAMILY_BY_PATTERN[name], windows, 0.6)
    if not found:
        return None
    # фигура, ради которой делали скрин, обычно крупная и ближе к правому краю
    target = n - max(3, int(0.12 * n))                 # обычно после фигуры на скрине ещё немного свечей
    best = max(found, key=lambda f: f["window"] + f.get("touches", 0) * 3
               - 1.5 * abs(f["start"] + f["window"] - target))
    return df, best
