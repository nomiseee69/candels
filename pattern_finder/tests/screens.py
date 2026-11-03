"""Рисует «скриншот TradingView» из свечей — для тестов чтения свечей со скрина."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def screenshot(df, path, theme="dark", volume=True, width=10, height=6):
    """Свечи (зелёные/красные), объём внизу отдельной панелью, сетка и шкала цены — как в TradingView."""
    bg = "#131722" if theme == "dark" else "white"
    grid = "#1e222d" if theme == "dark" else "#e0e3eb"
    n = len(df)
    fig = plt.figure(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.0, 0.25 if volume else 0.03, 0.93, 0.72 if volume else 0.94])
    for a in [ax]:
        a.set_facecolor(bg)
        a.grid(color=grid, linewidth=0.8)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    for i in range(n):
        color = "#089981" if c[i] >= o[i] else "#f23645"
        ax.vlines(i, lo[i], h[i], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), 1e-9), color=color))
    ax.set_xlim(-1, n)
    ax.set_xticks([])
    ax.tick_params(colors="#787b86", labelsize=8)
    ax.yaxis.tick_right()
    for s in ax.spines.values():
        s.set_visible(False)
    if volume and "volume" in df:
        vax = fig.add_axes([0.0, 0.02, 0.93, 0.18])
        vax.set_facecolor(bg)
        colors = ["#089981" if c[i] >= o[i] else "#f23645" for i in range(n)]
        vax.bar(range(n), df["volume"], color=colors, width=0.7, alpha=0.6)
        vax.set_xlim(-1, n)
        vax.axis("off")
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return str(path)
