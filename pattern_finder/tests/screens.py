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


def range_screenshot(df, start, end, path, poc=None, profile=True):
    """Скрин «как у пользователя»: серый фон, чёрные свечи, нарисованный прямоугольник боковика,
    белая линия POC и белый профиль объёма слева внутри прямоугольника."""
    n = len(df)
    fig = plt.figure(figsize=(8, 3), dpi=100)
    fig.patch.set_facecolor("#b2b5b8")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#b2b5b8")
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    for i in range(n):
        ax.vlines(i, lo[i], h[i], color="black", linewidth=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o[i], c[i])), 0.6, max(abs(c[i] - o[i]), 1e-9), color="black"))
    top, bottom = h[start:end + 1].max(), lo[start:end + 1].min()
    ax.add_patch(plt.Rectangle((start - 0.5, bottom), end - start + 1, top - bottom, fill=False,
                               edgecolor="black", linewidth=1.5))
    if poc is not None:
        ax.hlines(poc, start, start + (end - start) * 0.6, color="white", linewidth=1.5)
    if profile:
        levels = np.linspace(bottom, top, 30)
        weights = np.exp(-((levels - (poc if poc is not None else (top + bottom) / 2)) / (top - bottom) * 4) ** 2)
        for y, wgt in zip(levels, weights):
            ax.add_patch(plt.Rectangle((start, y), (end - start) * 0.15 * wgt, (top - bottom) / 40,
                                       color="white", alpha=0.9))
    ax.set_xlim(-2, n + 2)
    ax.set_ylim(lo.min() - (top - bottom) * 0.3, h.max() + (top - bottom) * 0.3)
    ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)
