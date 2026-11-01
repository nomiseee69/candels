"""История: все случаи паттерна на графике и что было после каждого.

Отвечает на вопрос «стоит ли доверять этому паттерну»: сколько раз он
встречался и куда потом пошла цена.
"""

import numpy as np

from shape import normalize, resample
from smc import find_all_fvg
from trendlines import KIND_NAMES, _analyze_window

HORIZON = 10               # исход смотрим через 10 свечей после паттерна
SHAPE_STEP = 2             # сдвиг окна при поиске форм (каждые 2 свечи)
LINE_STEP = 5              # сдвиг окна при поиске фигур из линий
FVG_FILL_WINDOW = 20       # «FVG отработал» = закрылся за 20 свечей


# --- общее ----------------------------------------------------------------------------------
def plural(n, one, few, many):
    """Русские окончания: 1 случай, 3 случая, 11 случаев."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


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
    for f in sorted(found, key=lambda f: f["score"], reverse=True):
        a0, a1 = f["start"], f["start"] + f["window"]
        if all(a1 <= k["start"] or a0 >= k["start"] + k["window"] for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f["start"])


# --- формы с картинки ---------------------------------------------------------------------------
def history_shapes(df, shape, windows, min_score):
    prices = df["close"].to_numpy()
    found = []
    for w in windows:
        for start in range(0, len(prices) - w + 1, SHAPE_STEP):
            segment = normalize(resample(prices[start:start + w], len(shape)))
            score = float(np.corrcoef(shape, segment)[0, 1])
            if score >= min_score:
                found.append({"start": start, "window": w, "score": score})
    found = non_overlapping(found)
    for f in found:
        f["outcome"] = outcome(df, f["start"] + f["window"] - 1)
    return found


# --- треугольники, клинья, каналы ------------------------------------------------------------------
def _breakout(df, found, horizon=HORIZON):
    """Куда цена вышла из фигуры за horizon свечей: "up", "down" или None."""
    x0, u0, x1, u1 = found["upper"]
    _, l0, _, l1 = found["lower"]
    up_slope = (u1 - u0) / (x1 - x0)
    lo_slope = (l1 - l0) / (x1 - x0)
    closes = df["close"].to_numpy()
    for x in range(int(x1) + 1, min(int(x1) + 1 + horizon, len(closes))):
        if closes[x] > u1 + up_slope * (x - x1):
            return "up"
        if closes[x] < l1 + lo_slope * (x - x1):
            return "down"
    return None


def history_lines(df, family, windows, min_score):
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    n = len(df)
    found = []
    for w in windows:
        for end in range(w, n + 1, LINE_STEP):
            f = _analyze_window(highs[end - w:end], lows[end - w:end])
            if f is None or f["kind"] not in family or f["score"] < min_score:
                continue
            start = end - w
            x0, u0, x1, u1 = f["upper"]
            _, l0, _, l1 = f["lower"]
            f.update({
                "start": start + x0,
                "window": w - x0,
                "upper": (start + x0, u0, start + x1, u1),
                "lower": (start + x0, l0, start + x1, l1),
            })
            found.append(f)
    found = non_overlapping(found)
    for f in found:
        end = f["start"] + f["window"] - 1
        f["outcome"] = outcome(df, end)
        f["breakout"] = _breakout(df, f) if end + 1 < n else None
    return found


# --- FVG ----------------------------------------------------------------------------------------------
def history_fvg(df):
    gaps = find_all_fvg(df, lookback=len(df), include_filled=True)
    for g in gaps:
        g["fill_candles"] = g["filled_at"] - g["end"] if g["filled_at"] is not None else None
    return sorted(gaps, key=lambda g: g["start"])


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
            line += f"\n   закрылись за {FVG_FILL_WINDOW} свечей: {len(quick)} из {len(judged)} ({len(quick) / len(judged):.0%})"
        return line

    line = f"• {name}: {plural(len(results), 'случай', 'случая', 'случаев')}"
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
        avg = np.mean([o["change"] for o in judged])
        line += (f"\n   через {HORIZON} свечей цена выше в {rose} из {len(judged)} "
                 f"({rose / len(judged):.0%}), в среднем {avg:+.1%}")
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
    return f"{'↑' if o['change'] > 0 else '↓'} {o['change']:+.1%}"


def describe_occurrence(df, family_kind, f):
    end = f["start"] + f["window"] - 1
    text = _date(df, end)
    if family_kind == "lines":
        text += f" {SHORT_KIND[f['kind']]}"
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
                where = "цена внутри" if g["distance"] == 0 else f"до зоны {g['distance']:.1%}"
                kind = "бычий" if g["kind"] == "bullish_fvg" else "медвежий"
                line += f"\n   ближайший открытый: {kind} {bottom:.6g}–{top:.6g}, {where}"
            lines.append(line)
            continue
        newest = sorted(found, key=lambda f: f["start"], reverse=True)
        shown = ", ".join(describe_occurrence(df, family_kind, f) for f in newest[:MAX_DATES])
        more = len(found) - MAX_DATES
        lines.append(f"• {name} ×{len(found)}: {shown}" + (f" и ещё {more}" if more > 0 else ""))
    return "\n".join(lines)
