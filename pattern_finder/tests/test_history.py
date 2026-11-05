import numpy as np
import pandas as pd

from history import (
    history_fvg, history_lines, history_shapes, non_overlapping, outcome, summarize,
)
from trendlines import TRIANGLES
from test_trendlines import make_df, zigzag

W_SHAPE = np.interp(np.linspace(0, 4, 50), [0, 1, 2, 3, 4], [1, 0, 0.6, 0.05, 1])


def plant_w(prices, at, length=40, depth=8):
    w = np.interp(np.linspace(0, 4, length), [0, 1, 2, 3, 4], [depth, 0, depth * 0.6, 0.4, depth])
    prices[at:at + length] = prices[at - 1] - depth + w
    return prices


def test_plural():
    from history import plural
    assert plural(1, "день", "дня", "дней") == "1 день"
    assert plural(83, "день", "дня", "дней") == "83 дня"
    assert plural(11, "день", "дня", "дней") == "11 дней"
    assert plural(25, "день", "дня", "дней") == "25 дней"


def test_outcome():
    df = make_df([100] * 5 + [101, 102, 103, 104, 105, 110] + [110] * 5)
    o = outcome(df, 4, horizon=6)
    assert abs(o["change"] - 0.10) < 1e-9
    assert outcome(df, len(df) - 3, horizon=6) is None   # слишком свежий — исход не известен


def test_non_overlapping_keeps_best():
    found = [
        {"start": 10, "window": 20, "score": 0.9},
        {"start": 12, "window": 20, "score": 0.95},   # почти то же место, но лучше
        {"start": 50, "window": 20, "score": 0.88},
    ]
    kept = non_overlapping(found)
    assert [f["start"] for f in kept] == [12, 50]


def test_history_finds_every_planted_w():
    rng = np.random.default_rng(4)
    prices = 100 + np.cumsum(rng.normal(0, 0.4, 400))
    for at in (60, 180, 300):
        prices = plant_w(prices, at)
    df = make_df(prices)
    found = history_shapes(df, W_SHAPE, windows=[40], min_score=0.9)
    starts = [f["start"] for f in found]
    for at in (60, 180, 300):
        assert any(abs(s - at) <= 2 for s in starts), (at, starts)
    assert all(f["outcome"] is not None for f in found)


def test_history_lines_with_breakout():
    rng = np.random.default_rng(5)
    before = 100 + np.cumsum(rng.normal(0, 0.3, 80))
    tri = zigzag(110, 103, 90, 99) + (before[-1] - 100)
    after = np.linspace(tri[-1], tri[-1] + 15, 20)                    # пробой вверх
    tail = after[-1] + np.cumsum(rng.normal(0, 0.3, 40))
    df = make_df(np.concatenate([before, tri, after, tail]))
    found = [f for f in history_lines(df, TRIANGLES, windows=(50,), min_score=0.85)
             if f["kind"] == "symmetric"]
    assert found
    assert found[-1]["breakout"] == "up"


def test_history_fvg_and_summary():
    rows = [(100, 100.5, 99.5, 100)] * 20 + [
        (100, 100.5, 99.5, 100.2), (100.2, 103.0, 100.1, 102.8), (102.8, 104.0, 102.0, 103.5),
        (103.5, 103.6, 100.2, 100.4),                                # закрыли сразу
    ] + [(100.4, 100.9, 99.9, 100.4)] * 30
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    gaps = [g for g in history_fvg(df) if g["kind"] == "bullish_fvg"]
    assert len(gaps) == 1 and gaps[0]["fill_candles"] == 1
    for g in gaps:
        g["_n"] = len(df)
    text = summarize("FVG", "fvg", gaps)
    assert "закрылись за 20 свечей: 1 из 1" in text and "%" not in text


def test_summary_shapes_text():
    results = [{"outcome": {"change": 0.02}}, {"outcome": {"change": -0.01}}, {"outcome": None}]
    text = summarize("Двойное дно", "shape", results)
    assert "3 случая" in text
    assert "выше в 1 из 2" in text and "%" not in text
