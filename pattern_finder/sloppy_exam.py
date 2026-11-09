"""Экзамен «неаккуратный трейдер» для Smart Money: зону и уровни рисуют НЕ точно — как в жизни.
Зона сдвинута на пару свечей, выше / ниже, больше / меньше, без заливки (одна рамка), рядом лишняя зона,
поверх стрелка и подпись «FVG», уровень BOS нарисован неровно от руки, короче, чуть мимо цены.
Бот должен всё равно понять, какой паттерн имел в виду человек.
Запуск:  python sloppy_exam.py            — все SMC-паттерны
         python sloppy_exam.py FVG BOS    — выбранные
"""

import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import screen_exam as E

SMC = ["FVG", "Order Block", "Breaker Block", "Mitigation Block", "BOS", "CHoCH", "EQH / EQL", "Range"]
ZONE_COLORS = ["#2962ff", "#ff9800", "#9c27b0", "#00bcd4"]


def render(df, path, theme, drawing, extras=(), volume=False, seed=0):
    """Как screen_exam.render, но с «неаккуратностями» поверх (extras)."""
    rng = np.random.default_rng(seed)
    bg, grid, text = E.THEMES[theme]
    n = len(df)
    fig = plt.figure(figsize=(9, 5), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.01, 0.06, 0.91, 0.92])
    ax.set_facecolor(bg)
    ax.grid(color=grid, linewidth=0.8)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    span = h.max() - lo.min()
    ax.set_ylim(lo.min() - span * 0.05, h.max() + span * 0.05)
    ax.set_xlim(-1, n + 6)
    for i in range(n):
        color = E.GREEN if c[i] >= o[i] else E.RED
        ax.vlines(i, lo[i], h[i], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), span * 0.002),
                                   color=color, linewidth=0))
    kind, args = drawing
    color = ZONE_COLORS[seed % len(ZONE_COLORS)]
    if kind == "zone":
        x0, bottom, top, fill = args
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), n + 5 - x0, top - bottom, facecolor=color if fill else "none",
                                   alpha=0.25 if fill else 1.0, edgecolor=color, linewidth=1.4))
    elif kind == "hline":
        x0, x1, y, wobble = args
        xs = np.linspace(x0, x1, 40)
        ys = y + (rng.normal(0, span * 0.004, len(xs)) if wobble else np.zeros(len(xs)))
        ax.plot(xs, ys, color=color, linewidth=2.2 if wobble else 1.6)
    elif kind == "box":
        x0, x1, bottom, top = args
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), x1 - x0 + 0.8, top - bottom, facecolor=color,
                                   alpha=0.15, edgecolor=color, linewidth=1.2))
    for extra in extras:
        if extra[0] == "zone2":                          # лишняя зона в другом месте
            _, x0, bottom, top = extra
            ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), 8, top - bottom, facecolor="#ff5252", alpha=0.2,
                                       edgecolor="#ff5252", linewidth=1))
        elif extra[0] == "arrow":                        # стрелка и подпись к паттерну
            _, x, y, label = extra
            ax.annotate(label, xy=(x, y), xytext=(x - 8, y + span * 0.18), color=text, fontsize=11,
                        arrowprops={"arrowstyle": "->", "color": text, "lw": 1.5})
    ax.yaxis.tick_right()
    ax.tick_params(colors=text, labelsize=8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path


def variants(drawing, df, item):
    """[(название, рисунок, лишнее)] — как разные люди рисуют тот же паттерн."""
    kind, args = drawing
    span = float(df["high"].max() - df["low"].min())
    atr = float((df["high"] - df["low"]).mean())
    out = []
    if kind == "zone":
        x0, bottom, top = args
        height = top - bottom
        mid = (top + bottom) / 2
        out = [
            ("точно", ("zone", (x0, bottom, top, True)), ()),
            ("на 2 свечи правее", ("zone", (x0 + 2, bottom, top, True)), ()),
            ("на 2 свечи левее", ("zone", (x0 - 2, bottom, top, True)), ()),
            ("выше на треть", ("zone", (x0, bottom + height / 3, top + height / 3, True)), ()),
            ("ниже на треть", ("zone", (x0, bottom - height / 3, top - height / 3, True)), ()),
            ("в 2 раза шире", ("zone", (x0, mid - height, mid + height, True)), ()),
            ("в 2 раза уже", ("zone", (x0, mid - height / 4, mid + height / 4, True)), ()),
            ("только рамка", ("zone", (x0, bottom, top, False)), ()),
            ("лишняя зона рядом", ("zone", (x0, bottom, top, True)),
             (("zone2", max(2, x0 - 20), bottom - 2 * height - atr, bottom - height - atr),)),
            ("стрелка и подпись", ("zone", (x0, bottom, top, True)), (("arrow", x0, top, "вот он"),)),
            ("без рисунка", (None, None), ()),
        ]
    elif kind == "hline":
        x0, x1, y = args
        out = [
            ("точно", ("hline", (x0, x1, y, False)), ()),
            ("от руки, неровно", ("hline", (x0, x1, y, True)), ()),
            ("короче", ("hline", (x0 + (x1 - x0) // 2, x1, y, False)), ()),
            ("чуть выше цены", ("hline", (x0, x1, y + 0.4 * atr, False)), ()),
            ("чуть ниже цены", ("hline", (x0, x1, y - 0.4 * atr, False)), ()),
            ("стрелка и подпись", ("hline", (x0, x1, y, False)), (("arrow", x1, y, "BOS"),)),
            ("без рисунка", (None, None), ()),
        ]
    elif kind == "box":
        x0, x1, bottom, top = args
        height = top - bottom
        out = [
            ("точно", ("box", (x0, x1, bottom, top)), ()),
            ("шире по времени", ("box", (x0 - 3, x1 + 3, bottom, top)), ()),
            ("выше и ниже краёв", ("box", (x0, x1, bottom - height * 0.15, top + height * 0.15)), ()),
            ("уже по цене", ("box", (x0, x1, bottom + height * 0.12, top - height * 0.12)), ()),
            ("стрелка и подпись", ("box", (x0, x1, bottom, top)), (("arrow", x1, top, "боковик"),)),
        ]
    return out


def run(patterns=SMC, per_pattern=3, seeds=(1, 2), say=print):
    folder = tempfile.mkdtemp()
    totals = {}
    for name in patterns:
        right = total = 0
        errors = []
        for seed in seeds:
            df = E.market(seed)
            for k, item in enumerate(E.cases(name, df, per_pattern)):
                start = int(item["start"])
                end = int(min(item.get("end", start + item.get("window", 1) - 1), len(df) - 1))
                lo, hi = start - E.BEFORE, min(len(df), end + 1 + E.AFTER)
                window_df = df.iloc[lo:hi].reset_index(drop=True)
                local = E._shift(item, lo)
                for v, (label, drawing, extras) in enumerate(variants(E._segment(item, lo), window_df, local)):
                    theme = "dark" if (seed + k + v) % 2 else "light"
                    path = render(window_df, os.path.join(folder, f"{seed}_{k}_{v}.png"), theme, drawing, extras,
                                  seed=seed + k + v)
                    try:
                        ok, seen = E.check(name, local, window_df, path)
                    except Exception as error:
                        ok, seen = False, f"ошибка: {error!r}"
                    right += ok
                    total += 1
                    t = totals.setdefault(label, [0, 0])
                    t[0] += ok
                    t[1] += 1
                    if not ok and len(errors) < 4:
                        errors.append(f"[{label}] {seen}")
        say(f"{name:18s} {right:3d}/{total:<3d} " + "; ".join(errors))
    say("\nПо видам неаккуратности:")
    for label, (r, t) in totals.items():
        say(f"  {label:22s} {r}/{t}")
    return totals


if __name__ == "__main__":
    run(sys.argv[1:] or SMC)
