import numpy as np
import pandas as pd

from trendlines import TRIANGLES, CHANNELS, WEDGES, find_line_pattern, kind_from_drawing
from vision import Line, analyze_screenshot


def make_df(closes, noise=0.15, seed=1):
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    wick = rng.uniform(0.05, noise, len(closes))
    return pd.DataFrame({
        "open": opens,
        "close": closes,
        "high": np.maximum(opens, closes) + wick,
        "low": np.minimum(opens, closes) - wick,
    })


def zigzag(top_start, top_end, bottom_start, bottom_end, n=60, waves=4):
    """Цена ходит между двумя прямыми — так выглядит фигура на графике."""
    t = np.linspace(0, 1, n)
    top = top_start + (top_end - top_start) * t
    bottom = bottom_start + (bottom_end - bottom_start) * t
    wave = (np.sin(np.linspace(0, waves * 2 * np.pi, n)) + 1) / 2
    return bottom + (top - bottom) * wave


def with_history(pattern, history=80, seed=2):
    rng = np.random.default_rng(seed)
    before = pattern[0] + np.cumsum(rng.normal(0, 0.8, history))[::-1] * 0.3
    return np.concatenate([before, pattern])


# --- рынок ---------------------------------------------------------------------------------
def test_symmetric_triangle_found():
    df = make_df(with_history(zigzag(110, 103, 90, 99)))
    found = find_line_pattern(df, TRIANGLES)
    assert found is not None
    assert found["kind"] == "symmetric"
    assert found["score"] > 0.8
    assert found["start"] + found["window"] == len(df)


def test_flat_tops_still_found():
    """Регрессия: соседние свечи с одинаковым high («плоская вершина») не должны
    прятать экстремум. Без фитилей-случайностей high = max(open, close) + 0.1."""
    closes = with_history(zigzag(110, 103, 90, 99))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    df = pd.DataFrame({
        "open": opens, "close": closes,
        "high": np.maximum(opens, closes) + 0.1, "low": np.minimum(opens, closes) - 0.1,
    })
    found = find_line_pattern(df, TRIANGLES)
    assert found is not None and found["kind"] == "symmetric"


def test_ascending_triangle_found():
    df = make_df(with_history(zigzag(110, 110, 90, 105)))
    found = find_line_pattern(df, TRIANGLES)
    assert found is not None and found["kind"] == "ascending"


def test_falling_wedge_is_not_a_triangle():
    df = make_df(with_history(zigzag(110, 95, 100, 92)))
    assert find_line_pattern(df, TRIANGLES) is None
    found = find_line_pattern(df, WEDGES)
    assert found is not None and found["kind"] == "falling_wedge"


def test_channel_found():
    df = make_df(with_history(zigzag(110, 120, 100, 110)))
    found = find_line_pattern(df, CHANNELS)
    assert found is not None and found["kind"] == "channel_up"


def test_breakout_is_not_reported():
    prices = list(with_history(zigzag(110, 103, 90, 99)))
    prices[-1] = 120                    # пробой вверх — фигура уже отработала
    df = make_df(prices)
    assert find_line_pattern(df, TRIANGLES) is None


def test_random_walk_rarely_matches():
    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        df = make_df(100 + np.cumsum(rng.normal(0, 1, 200)), seed=seed)
        found = find_line_pattern(df, TRIANGLES)
        if found and found["score"] > 0.85:
            hits += 1
    assert hits <= 5


# --- рисунок пользователя ---------------------------------------------------------------------
def test_drawing_symmetric():
    lines = [Line(0, 100, 300, 160), Line(0, 300, 300, 200)]   # верх вниз, низ вверх
    assert kind_from_drawing(lines) == "symmetric"


def test_drawing_ascending():
    lines = [Line(0, 100, 300, 100), Line(0, 300, 300, 140)]
    assert kind_from_drawing(lines) == "ascending"


def test_drawing_channel():
    lines = [Line(0, 100, 300, 40), Line(0, 200, 300, 140)]
    assert kind_from_drawing(lines) == "channel_up"


def test_user_screenshot_is_symmetric_triangle(user_triangle):
    a = analyze_screenshot(user_triangle)
    assert kind_from_drawing(a.lines) == "symmetric"
