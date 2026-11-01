"""Общие тестовые картинки. Синтетические графики рисуются на лету во временной папке."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, ImageDraw

SAMPLES = Path(__file__).parent / "samples"


def _candles_chart(path, theme, line_color, seed):
    rng = np.random.default_rng(seed)
    n = 80
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    closes[30:60] = 100 - 6 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 30))) + rng.normal(0, 0.3, 30)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + rng.uniform(0.2, 1.0, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.2, 1.0, n)

    bg = "white" if theme == "light" else "#131722"
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.grid(color="#e0e3eb" if theme == "light" else "#1e222d", linewidth=0.8)
    for i in range(n):
        color = "#089981" if closes[i] >= opens[i] else "#f23645"
        ax.vlines(i, lows[i], highs[i], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.3, min(opens[i], closes[i])), 0.6,
                                   abs(closes[i] - opens[i]) + 0.05, color=color))
    ax.plot([25, 70], [107, 99], color=line_color, linewidth=2)   # верхняя линия вниз
    ax.plot([30, 75], [92, 100], color=line_color, linewidth=2)   # нижняя линия вверх
    ax.set_xlim(-1, n)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return str(path)


@pytest.fixture
def light_candles(tmp_path):
    return _candles_chart(tmp_path / "light.png", "light", "black", seed=3)


@pytest.fixture
def dark_candles_blue(tmp_path):
    return _candles_chart(tmp_path / "dark.png", "dark", "#2962ff", seed=5)


@pytest.fixture
def w_line(tmp_path):
    path = tmp_path / "w.png"
    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    d.line([(20, 40), (110, 160), (200, 80), (290, 150), (380, 30)], fill="black", width=3)
    img.save(path)
    return str(path)


@pytest.fixture
def user_triangle():
    return str(SAMPLES / "dark_triangle.png")


@pytest.fixture
def user_zigzag():
    return str(SAMPLES / "dark_avax_lines.png")


@pytest.fixture
def user_noisy():
    return str(SAMPLES / "dark_avax_labels.png")
