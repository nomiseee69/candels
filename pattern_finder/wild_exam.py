"""Экзамен «скрины как в жизни»: те же случаи, что в screen_exam, но картинка выглядит так,
как её присылают РАЗНЫЕ люди из разных программ:
  • другие цвета свечей (Binance, синие / красные, белые / серые, чёрно-белые, пустые «hollow»);
  • интерфейс вокруг графика: подписи сверху, шкала цены с цветной плашкой последней цены,
    пунктир последней цены, шкала времени, водяной знак, панель инструментов слева;
  • вертикальный скрин с телефона, очень много мелких свечей;
  • рисунок по-разному: зона с подписью внутри («FVG», «OB»), короткая зона (не до края),
    зона пунктирной рамкой, уровень пунктиром, уровень с кружками на концах, толстая линия.
Запуск:  python wild_exam.py            — все паттерны
         python wild_exam.py FVG BOS    — выбранные
"""

import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import screen_exam as E
from catalog import ALL_PATTERNS, LINE_PATTERNS, SHAPE_PATTERNS

LABEL = {"FVG": "FVG", "Order Block": "OB", "Breaker Block": "Breaker", "Mitigation Block": "MB",
         "BOS": "BOS", "CHoCH": "CHoCH", "EQH / EQL": "EQL", "Range": "Range"}

# вариант: название → настройки картинки
VARIANTS = {
    "цвета Binance": dict(theme="dark", up="#0ecb81", down="#f6465d"),
    "синие и красные": dict(theme="light", up="#2962ff", down="#f23645"),
    "белые и серые": dict(theme="dark", up="#ffffff", down="#787b86"),
    "чёрно-белые": dict(theme="light", up="#ffffff", down="#000000", edge="#000000"),
    "пустые (hollow)": dict(theme="dark", hollow=True),
    "интерфейс TradingView": dict(theme="dark", ui=True, volume=True),
    "интерфейс, светлая": dict(theme="light", ui=True),
    "телефон вертикально": dict(theme="dark", ui=True, size=(4.6, 8.2)),
    "много мелких свечей": dict(theme="dark", before=110, volume=True),
    "подпись в зоне": dict(theme="light", text=True),
    "короткий рисунок": dict(theme="dark", short=True),
    "пунктир": dict(theme="light", dashed=True),
    "кружки и толстая линия": dict(theme="dark", handles=True),
}


def render(df, path, name, drawing, theme="dark", up=E.GREEN, down=E.RED, edge=None, hollow=False, ui=False,
           volume=False, size=(9, 5), dpi=100, text=False, short=False, dashed=False, handles=False, **_):
    bg, grid, ink = E.THEMES[theme]
    n = len(df)
    fig = plt.figure(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(bg)
    box = [0.06, 0.1, 0.82, 0.8] if ui else [0.01, 0.06, 0.91, 0.92]
    ax = fig.add_axes(box)
    ax.set_facecolor(bg)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)                          # как в TradingView: свечи поверх сетки
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    span = h.max() - lo.min()
    ax.set_ylim(lo.min() - span * (0.25 if volume else 0.05), h.max() + span * 0.05)
    ax.set_xlim(-1, n + 6)
    if volume and "volume" in df:
        v = df["volume"].to_numpy()
        scale = span * 0.18 / v.max()
        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.35, lo.min() - span * 0.25), 0.7, v[i] * scale,
                                       color=up if c[i] >= o[i] else down, alpha=0.35, linewidth=0))
    for i in range(n):
        rising = c[i] >= o[i]
        color = up if rising else down
        line = edge or color
        body = max(abs(c[i] - o[i]), span * 0.002)
        if hollow and rising:                       # пустая свеча: только контур, фитиль не идёт сквозь тело
            ax.vlines(i, max(o[i], c[i]), h[i], color=color, linewidth=1)
            ax.vlines(i, lo[i], min(o[i], c[i]), color=color, linewidth=1)
            ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, body, facecolor=bg, edgecolor=color,
                                       linewidth=1))
            continue
        ax.vlines(i, lo[i], h[i], color=line, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, body, facecolor=color,
                                   edgecolor=edge, linewidth=0.8 if edge else 0))
    kind, args = drawing
    style = "--" if dashed else "-"
    if kind == "zone":
        x0, bottom, top = args
        width = min(12, n + 5 - x0) if short else n + 5 - x0
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), width, top - bottom, facecolor=E.DRAW["zone"],
                                   alpha=0.2, edgecolor=E.DRAW["zone"], linewidth=1.2, linestyle=style))
        if text:
            ax.text(x0 + width / 2, (top + bottom) / 2, LABEL.get(name, name), color=E.DRAW["zone"],
                    fontsize=9, ha="center", va="center")
    elif kind == "hline":
        x0, x1, y = args
        if short:
            x0 = x0 + (x1 - x0) // 3
        ax.plot([x0, x1], [y, y], color=E.DRAW["line"], linewidth=2.6 if handles else 1.6, linestyle=style,
                marker="o" if handles else None, markersize=6, markerfacecolor=bg)
        if text:
            ax.text((x0 + x1) / 2, y + span * 0.012, LABEL.get(name, name), color=E.DRAW["line"], fontsize=9,
                    ha="center", va="bottom")
    elif kind == "box":
        x0, x1, bottom, top = args
        ax.add_patch(plt.Rectangle((x0 - 0.4, bottom), x1 - x0 + 0.8, top - bottom, facecolor=E.DRAW["box"],
                                   alpha=0.15, edgecolor=E.DRAW["box"], linewidth=1.2, linestyle=style))
        if text:
            ax.text((x0 + x1) / 2, top - (top - bottom) * 0.12, LABEL.get(name, name), color=E.DRAW["box"],
                    fontsize=9, ha="center", va="center")
    elif kind == "lines":
        for x0, y0, x1, y1 in args:
            ax.plot([x0, x1], [y0, y1], color=E.DRAW["line"], linewidth=2.8 if handles else 1.8, linestyle=style,
                    marker="o" if handles else None, markersize=6, markerfacecolor=bg)
    ax.yaxis.tick_right()
    ax.tick_params(colors=ink, labelsize=8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if ui:                                          # то, что окружает график в TradingView / на бирже
        last = c[-1]
        ax.axhline(last, color=up if c[-1] >= o[-1] else down, linewidth=0.8, linestyle=":")
        ax.text(1.005, last, f"{last:.2f}", transform=ax.get_yaxis_transform(), color="#ffffff", fontsize=8,
                va="center", bbox=dict(facecolor=up if c[-1] >= o[-1] else down, edgecolor="none", pad=2))
        fig.text(box[0] + 0.01, 0.865, "BTCUSDT · 4h · BINANCE", color=ink, fontsize=10)
        fig.text(box[0] + 0.01, 0.835, f"O {o[-1]:.2f} H {h[-1]:.2f} L {lo[-1]:.2f} C {c[-1]:.2f} +1.24%",
                 color=up, fontsize=8)
        fig.text(box[0] + 0.01, 0.808, "Vol · BTC 12.4K", color=down, fontsize=8)
        for k, label in enumerate(["12", "14", "16", "18", "20", "22", "24"]):
            fig.text(box[0] + 0.1 + k * 0.105, 0.07, label, color=ink, fontsize=8)
        fig.text(box[0] + 0.3, 0.45, "BTCUSDT", color=grid, fontsize=34, alpha=0.9)
        fig.add_artist(plt.Rectangle((0, 0.93), 1, 0.07, facecolor=grid, edgecolor="none"))    # верхняя панель
        fig.text(0.02, 0.955, "BTCUSDT   4h   Indicators   Alert   Replay", color=ink, fontsize=9)
        fig.add_artist(plt.Rectangle((0, 0), 0.045, 0.93, facecolor=grid, edgecolor="none"))   # инструменты слева
        for k in range(9):
            fig.add_artist(plt.Rectangle((0.012, 0.84 - k * 0.085), 0.02, 0.035, facecolor="none",
                                         edgecolor=ink, linewidth=1))
        fig.text(0.06, 0.02, "1D  5D  1M  3M  6M  YTD  1Y  All", color=ink, fontsize=8)
    fig.savefig(path, facecolor=bg, dpi=dpi)
    plt.close(fig)
    return path


def cases_for(name, per_pattern, seeds, before):
    result = []
    if name in SHAPE_PATTERNS:
        for k in range(per_pattern * len(seeds)):
            window_df, key = E.planted(name, 100 + k)
            result.append((window_df, {"key": key}, (None, None)))
        return result
    for seed in seeds:
        df = E.market(seed)
        for k, item in enumerate(E.cases(name, df, per_pattern)):
            start = int(item["start"])
            end = int(min(item.get("end", start + item.get("window", 1) - 1), len(df) - 1))
            lo, hi = max(0, start - before), min(len(df), end + 1 + E.AFTER)
            drawing = E._segment(item, lo)
            if name in LINE_PATTERNS and k % 2:
                drawing = (None, None)
            result.append((df.iloc[lo:hi].reset_index(drop=True), E._shift(item, lo), drawing))
    return result


def run(patterns=ALL_PATTERNS, per_pattern=3, seeds=(1, 2), say=print, only=None, keep=None):
    folder = keep or tempfile.mkdtemp()
    totals = {label: [0, 0] for label in VARIANTS}
    table = {}
    for name in patterns:
        right = total = 0
        errors = []
        for label, options in VARIANTS.items():
            if only and label not in only:
                continue
            cases = cases_for(name, per_pattern, seeds, options.get("before", E.BEFORE))
            for i, (window_df, item, drawing) in enumerate(cases):
                path = os.path.join(folder, f"{abs(hash((name, label))) % 10 ** 8}_{i}.png")
                render(window_df, path, name, drawing, **options)
                try:
                    ok, seen = E.check(name, item, window_df, path)
                except Exception as error:
                    ok, seen = False, f"ошибка: {error!r}"
                right += ok
                total += 1
                totals[label][0] += ok
                totals[label][1] += 1
                if not ok:
                    errors.append(f"[{label}] {seen}")
                    if keep:
                        say(f"   ✗ {name} [{label}] #{i}: {seen} → {path}")
        table[name] = (right, total, errors)
        say(f"{name:18s} {right:3d}/{total:<3d} " + "; ".join(errors[:4]))
    say("\nПо видам скринов:")
    for label, (r, t) in totals.items():
        if t:
            say(f"  {label:26s} {r}/{t}")
    return table, totals


if __name__ == "__main__":
    run(sys.argv[1:] or ALL_PATTERNS)
