"""«Похоже на твой пример»: найденная фигура против рисунка пользователя."""

from pathlib import Path

import scanner
from similarity import compare, figure_shape, reference_from_screenshot
from test_trendlines import make_df, screenshot_candles, with_history, zigzag
from trendlines import CHANNELS, TRIANGLES, _analyze_window, find_line_pattern

SAMPLES = Path(__file__).parent / "samples"
CHANNEL = str(SAMPLES / "dark_channel_down.png")


def channel_df(top0, top1, bottom0, bottom1):
    return make_df(with_history(zigzag(top0, top1, bottom0, bottom1, n=80, waves=4)))


def test_reference_from_users_drawing():
    ref = reference_from_screenshot(CHANNEL)
    assert ref is not None and ref["path"] is not None
    u0, u1, l0, l1 = ref["geometry"]
    assert u0 == 1.0 and l1 == 0.0            # вписано в квадрат: верх слева, низ справа
    assert u0 > u1 and l0 > l1                # обе линии идут вниз — нисходящий канал


def test_same_chart_is_almost_identical():
    ref = reference_from_screenshot(CHANNEL)
    _, _, df = screenshot_candles(CHANNEL)
    found = _analyze_window(df["high"].tolist(), df["low"].tolist(), df["close"].tolist())
    assert compare(ref, figure_shape(df, found)) >= 0.9


def test_similar_and_different_figures():
    ref = reference_from_screenshot(CHANNEL)
    down = channel_df(110, 95, 100, 85)
    up = channel_df(100, 115, 90, 105)
    tri = make_df(with_history(zigzag(110, 103, 90, 99, n=80, waves=4)))
    sim_down = compare(ref, figure_shape(down, find_line_pattern(down, CHANNELS)))
    sim_up = compare(ref, figure_shape(up, find_line_pattern(up, CHANNELS)))
    sim_tri = compare(ref, figure_shape(tri, find_line_pattern(tri, TRIANGLES)))
    assert sim_down >= 0.8
    assert sim_up < 0.3 and sim_tri < 0.4      # другой наклон / сужение — непохоже


def test_scanner_reports_and_filters_by_similarity():
    ref = reference_from_screenshot(CHANNEL)
    down, up = channel_df(110, 95, 100, 85), channel_df(100, 115, 90, 105)
    found = scanner.find_lines(down, "Канал", "channel_down", [ref])
    assert found is not None and found["similarity"] >= 0.8
    assert "похоже на твой пример" in scanner.describe_item({**found, "name": "Канал"})
    assert scanner.find_lines(up, "Канал", None, [ref]) is not None             # без фильтра — есть
    assert scanner.find_lines(up, "Канал", None, [ref], similar_only=True) is None   # непохожий отсеян
