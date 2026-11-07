"""Экзамен «понимает ли бот скриншоты» — по КАЖДОМУ паттерну.

Как проверяем (без людей и без интернета):
  1) на графике находим настоящие случаи паттерна правилами бота — это «правильный ответ»;
  2) рисуем участок «как скриншот TradingView»: свечи, сетка, шкала цены, светлая или тёмная
     тема, и то, что обычно рисует трейдер: закрашенную зону (FVG, OB…), линию уровня (BOS,
     CHoCH, EQH/EQL), прямоугольник (Range), две линии (треугольник, клин, канал, флаг);
  3) отдаём картинку боту ровно тем путём, каким идёт твой скрин, и сверяем ответ:
     тот ли паттерн, то ли направление, то ли место.
Итог — таблица «паттерн → сколько скринов понял правильно». Запуск:  python screen_exam.py
"""

import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catalog import ALL_PATTERNS, FLAG_PATTERNS, FLAG_SHAPE, LINE_PATTERNS, SHAPE_PATTERNS

BEFORE, AFTER = 35, 10          # сколько свечей до и после паттерна на «скрине»
THEMES = {"dark": ("#131722", "#1e222d", "#d1d4dc"), "light": ("#ffffff", "#f0f3fa", "#131722")}
GREEN, RED = "#089981", "#f23645"
DRAW = {"zone": "#9c27b0", "line": "#2962ff", "box": "#2962ff"}
LOCATION = 4                    # «то же место» — не дальше 4 свечей


# --- рынок для экзамена ---------------------------------------------------------------------------
def market(seed, n=1200):
    """Рынок с трендами, импульсами и боковиками (не чистое случайное блуждание)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.004, n // 40 + 1), 40)[:n]
    vol = np.repeat(rng.uniform(0.006, 0.02, n // 60 + 1), 60)[:n]
    shocks = rng.standard_t(3, n) * vol + drift
    jumps = rng.random(n) < 0.02
    shocks[jumps] += rng.choice([-1, 1], jumps.sum()) * rng.uniform(0.03, 0.06, jumps.sum())
    c = 100 * np.exp(np.cumsum(shocks))
    o = np.roll(c, 1)
    o[0] = c[0]
    o = o + (c - o) * rng.uniform(-0.1, 0.1, n)
    wick = np.abs(rng.normal(0, 0.5, n)) * vol * c
    v = rng.lognormal(0, 0.4, n) * (1 + 4 * jumps)
    return pd.DataFrame({"open_time": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
                         "open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick,
                         "close": c, "volume": v})


SHAPE_PLAN = {   # опорные точки формы: (через сколько свечей, цена); перед ними — 20 свечей шума
    "Двойное дно": [(0, 112), (16, 100), (9, 106.5), (9, 100.3), (9, 109), (6, 110)],
    "Голова и плечи": [(0, 90), (14, 100.5), (7, 96), (8, 105.5), (8, 96.4), (7, 100.8), (9, 92), (5, 91)],
}
KEY_POINT = {"Двойное дно": 3, "Голова и плечи": 2}      # какую точку фигуры проверяем: второе дно / голова
KEY_PLAN = {"Двойное дно": 3, "Голова и плечи": 3}       # та же точка в SHAPE_PLAN


def planted(name, seed):
    """Скрин с нарисованной (подложенной) фигурой: (свечи, номер ключевой точки)."""
    rng = np.random.default_rng(seed)
    base = {"Двойная вершина": "Двойное дно", "Перевёрнутая ГиП": "Голова и плечи"}.get(name, name)
    plan = SHAPE_PLAN[base]
    scale = rng.uniform(0.7, 1.3)
    xs, ys, x = [], [], 20
    for step, price in plan:
        x += step
        xs.append(x)
        ys.append(100 + (price - 100) * scale)
    n = xs[-1] + 1
    path = np.interp(np.arange(n), xs, ys)
    path[:20] = ys[0] + np.cumsum(rng.normal(0, 0.4, 20))[::-1] - np.cumsum(rng.normal(0, 0.4, 20))[-1]
    path[:20] += ys[0] - path[19]
    c = path + rng.normal(0, 0.35, n)
    if name != base:                                   # вершина = дно вверх ногами
        c = 200 - c
    o = np.roll(c, 1)
    o[0] = c[0]
    wick = np.abs(rng.normal(0, 0.35, n))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c,
                       "volume": rng.lognormal(0, 0.3, n)})
    return df, xs[KEY_PLAN[base]]


def cases(name, df, limit):
    """Настоящие случаи паттерна на графике — правилами бота. [(находка)]"""
    from scanner import build_history_jobs
    from shapes_check import validate
    if name in SHAPE_PATTERNS:
        from history import history_shapes
        from scanner import MIN_SCORE, WINDOWS
        from templates import template_shape
        found = [f for f in history_shapes(df, template_shape(name), WINDOWS, MIN_SCORE, name) if f.get("points")]
    else:
        jobs, _ = build_history_jobs([{"name": name}], {"fvg": "all", "ob_volume": False})   # чтение — на всех блоках
        found = [f for _, _, run in jobs for f in run(df)]
    n = len(df)
    result = []
    for f in found:
        start = int(f["start"])
        end = int(min(f.get("end", start + f.get("window", 1) - 1), n - 1))
        if start < BEFORE or end + AFTER >= n:
            continue
        if name == "FVG" and not f.get("significant"):
            continue                                   # на скрин трейдер ставит заметный FVG
        if name in ("BOS", "CHoCH") and end - start < 5:
            continue
        result.append(f)
    step = max(1, len(result) // limit)
    return result[::step][:limit]


# --- «скриншот» ---------------------------------------------------------------------------------
def _segment(item, lo):
    """Что нарисовал бы трейдер: (тип, параметры) в координатах свечей скрина."""
    kind = str(item.get("kind", ""))
    if "zone" in item:
        bottom, top = item["zone"]
        return ("zone", (item["start"] - lo, bottom, top))
    if kind in ("bos", "choch"):
        return ("hline", (item["start"] - lo, item["end"] - lo, item["level"]))
    if kind in ("eqh", "eql"):
        return ("hline", (item["touches"][0] - lo, item["end"] - lo + 3, item["level"]))
    if kind == "range" or "box" in item:
        bottom, top = item["box"]
        return ("box", (item["start"] - lo, item["end"] - lo, bottom, top))
    if "upper" in item:
        shift = lambda ln: (ln[0] - lo, ln[1], ln[2] - lo, ln[3])
        return ("lines", (shift(item["upper"]), shift(item["lower"])))
    return (None, None)


def render(df, path, theme="dark", drawing=(None, None), volume=True, width=9, height=5):
    bg, grid, text = THEMES[theme]
    n = len(df)
    fig = plt.figure(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.01, 0.06, 0.91, 0.92])
    ax.set_facecolor(bg)
    ax.grid(color=grid, linewidth=0.8)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    span = h.max() - lo.min()
    ax.set_ylim(lo.min() - span * (0.25 if volume else 0.05), h.max() + span * 0.05)
    ax.set_xlim(-1, n + 6)
    if volume and "volume" in df:                     # объём — внизу той же панели, как в TradingView
        v = df["volume"].to_numpy()
        scale = span * 0.18 / v.max()
        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.35, lo.min() - span * 0.25), 0.7, v[i] * scale,
                                       color=GREEN if c[i] >= o[i] else RED, alpha=0.35, linewidth=0))
    for i in range(n):
        color = GREEN if c[i] >= o[i] else RED
        ax.vlines(i, lo[i], h[i], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), span * 0.002),
                                   color=color, linewidth=0))
    kind, args = drawing
    if kind == "zone":
        x0, bottom, top = args
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), n + 5 - x0, top - bottom, facecolor=DRAW["zone"],
                                   alpha=0.2, edgecolor=DRAW["zone"], linewidth=1.2))
    elif kind == "hline":
        x0, x1, y = args
        ax.hlines(y, x0, x1, color=DRAW["line"], linewidth=1.6)
    elif kind == "box":
        x0, x1, bottom, top = args
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), x1 - x0 + 0.8, top - bottom, facecolor=DRAW["box"],
                                   alpha=0.15, edgecolor=DRAW["box"], linewidth=1.2))
    elif kind == "lines":
        for x0, y0, x1, y1 in args:
            ax.plot([x0, x1], [y0, y1], color=DRAW["line"], linewidth=1.8)
    ax.yaxis.tick_right()
    ax.tick_params(colors=text, labelsize=8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path


# --- что бот понял --------------------------------------------------------------------------------
def _same_direction(a, b):
    from quality import is_bullish
    return is_bullish(a) == is_bullish(b)


def check(name, item, window_df, path):
    """Бот читает скрин тем же путём, что и твой. Возвращает (понял правильно?, что увидел)."""
    from fingerprint import FAMILY, screenshot_fingerprint, range_from_box
    from vision import analyze_screenshot, extract_marks, extract_range
    lo = 0
    if name in SHAPE_PATTERNS:
        from shapes_check import read_shape
        found = read_shape(name, path)
        if found is None:
            return False, "фигуру по точкам не нашёл"
        candles, shape = found
        points = [x for x, _ in shape["points"]]
        key = KEY_POINT[{"Двойная вершина": "Двойное дно", "Перевёрнутая ГиП": "Голова и плечи"}.get(name, name)]
        shift = len(candles) - len(window_df)
        ok = abs(points[key] - item["key"]) <= LOCATION + abs(shift)
        return ok, f"{len(candles)}/{len(window_df)} свечей, ключевая точка у {points[key]} (верно {item['key']})"
    if name in LINE_PATTERNS:
        from history import read_lines_figure
        from trendlines import kind_from_drawing
        analysis = analyze_screenshot(path)
        figure = kind_from_drawing(analysis.lines)
        how = f"{len(analysis.lines)} линий"
        if figure is None:
            seen = read_lines_figure(name, path)
            figure, how = (seen[1]["kind"], "по свечам") if seen else (None, "по свечам не нашёл")
        return figure == item["kind"], f"увидел {figure} ({how})"
    from bot import drawn_levels
    if name == "Range":
        candles, box = extract_range(path)
        if candles is None:
            return False, "не прочитал свечи"
        if box is None:
            return False, f"{len(candles)} свечей, прямоугольник не нашёл"
        ok = abs(box["start"] - item["start"]) <= LOCATION + 2 and abs(box["end"] - item["end"]) <= LOCATION + 2
        return ok, f"боковик {box['start']}–{box['end']} (верно {item['start']}–{item['end']})"
    candles, zones, lines = extract_marks(path)
    if candles is None:
        return False, "не прочитал свечи"
    fp, seen = screenshot_fingerprint(name, candles, drawn_levels(path), zones, lines)
    if seen is None:
        return False, f"{len(candles)} свечей, паттерн не нашёл"
    shift = len(candles) - len(window_df)                 # потерянные свечи слева/справа — допуск
    where = abs(int(seen["start"]) - int(item["start"])) <= LOCATION + abs(shift)
    if name in FLAG_PATTERNS:
        where = abs(int(seen.get("end", seen["start"])) - int(item.get("end", item["start"]))) <= LOCATION + 2 + abs(shift)
    same = _same_direction(seen, item) and str(seen.get("kind")) == str(item.get("kind"))
    return same and where, (f"{len(candles)}/{len(window_df)} свечей, увидел {seen.get('kind')} у свечи "
                            f"{seen['start']} (верно {item.get('kind')} у {item['start']})")


def run(patterns=ALL_PATTERNS, per_pattern=6, seeds=(1, 2, 3), say=print):
    """Таблица: паттерн → (правильно, всего, примеры ошибок)."""
    table = {}
    folder = tempfile.mkdtemp()
    for name in patterns:
        right, total, errors = 0, 0, []
        if name in SHAPE_PATTERNS:                      # фигуры подкладываем сами — их на рынке мало
            for k in range(per_pattern * len(seeds)):
                window_df, key = planted(name, 100 + k)
                theme = "dark" if k % 2 else "light"
                # половина скринов — без объёма: свечи тогда доходят до самого низа картинки
                path = render(window_df, os.path.join(folder, f"shape_{k}.png"), theme, volume=k % 4 < 2)
                try:
                    ok, seen = check(name, {"key": key}, window_df, path)
                except Exception as error:
                    ok, seen = False, f"ошибка: {error!r}"
                right += ok
                total += 1
                if not ok and len(errors) < 3:
                    errors.append(f"{theme}: {seen}")
            table[name] = (right, total, errors)
            say(f"{name:18s} {right:2d}/{total:<2d} " + ("; ".join(errors) if errors else ""))
            continue
        for seed in seeds:
            df = market(seed)
            for k, item in enumerate(cases(name, df, per_pattern)):
                start = int(item["start"])
                end = int(min(item.get("end", start + item.get("window", 1) - 1), len(df) - 1))
                lo, hi = start - BEFORE, min(len(df), end + 1 + AFTER)
                window_df = df.iloc[lo:hi].reset_index(drop=True)
                local = _shift(item, lo)
                drawing = _segment(item, lo)
                if name in LINE_PATTERNS and k % 2:
                    drawing = (None, None)                   # половину фигур — без нарисованных линий
                theme = "dark" if (seed + k) % 2 else "light"
                path = render(window_df, os.path.join(folder, f"{seed}_{k}.png"), theme, drawing,
                              volume=k % 4 < 2)
                try:
                    ok, seen = check(name, local, window_df, path)
                except Exception as error:                 # экзамен не должен падать
                    ok, seen = False, f"ошибка: {error!r}"
                right += ok
                total += 1
                if not ok and len(errors) < 3:
                    errors.append(f"{theme}: {seen}")
        table[name] = (right, total, errors)
        say(f"{name:18s} {right:2d}/{total:<2d} " + ("; ".join(errors) if errors else ""))
    return table


def _shift(item, lo):
    """Находка в координатах скрина (номера свечей с начала скрина)."""
    out = dict(item)
    for key in ("start", "end", "touched_at", "broken_at", "filled_at", "swept_at", "swing"):
        if out.get(key) is not None:
            out[key] = int(out[key]) - lo
    if "touches" in out and isinstance(out["touches"], list):
        out["touches"] = [t - lo for t in out["touches"]]
    for key in ("upper", "lower", "pole"):
        if key in out:
            x0, y0, x1, y1 = out[key]
            out[key] = (x0 - lo, y0, x1 - lo, y1)
    if out.get("points"):
        out["points"] = [(x - lo, y) for x, y in out["points"]]
    return out


if __name__ == "__main__":
    names = [a for a in sys.argv[1:]] or ALL_PATTERNS
    run(names)
