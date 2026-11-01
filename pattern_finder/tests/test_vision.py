import numpy as np

from extremes import find_local_minima
from shape import normalize, photo_to_shape, resample
from vision import Line, analyze_screenshot, classify_pair, describe_lines


# --- распознавание скриншотов ------------------------------------------------------
def test_light_theme_black_lines(light_candles):
    a = analyze_screenshot(light_candles)
    assert a.mode == "candles"
    assert a.ok
    angles = sorted(round(ln.angle) for ln in a.lines)
    assert len(angles) == 2
    assert angles[0] < -10 and angles[1] > 10
    assert "сходятся" in describe_lines(a.lines)


def test_dark_theme_blue_lines(dark_candles_blue):
    a = analyze_screenshot(dark_candles_blue)
    assert a.mode == "candles"
    assert len(a.lines) == 2


def test_user_triangle_screenshot(user_triangle):
    a = analyze_screenshot(user_triangle)
    assert a.mode == "candles"
    angles = sorted(ln.angle for ln in a.lines)
    assert len(angles) == 2
    assert -16 < angles[0] < -8        # верхняя линия слегка вниз
    assert 48 < angles[1] < 60         # нижняя линия круто вверх
    assert "треугольник" in describe_lines(a.lines)


def test_user_zigzag_becomes_pattern(user_zigzag):
    a = analyze_screenshot(user_zigzag)
    assert a.chain is not None
    assert len(a.chain) == 5           # 4 отрезка → 5 вершин ломаной W
    shape = normalize(resample(a.pattern_series(), 50))
    # у W два дна
    assert len(find_local_minima(list(shape), 3)) == 2


def test_noisy_screenshot_still_reads_price(user_noisy):
    a = analyze_screenshot(user_noisy)
    assert a.mode == "candles"
    assert a.ok
    # цена не «проваливается» в объём внизу картинки
    assert a.prices.min() > a.height * 0.25


def test_line_chart_w(w_line):
    a = analyze_screenshot(w_line)
    assert a.mode == "line"
    assert a.lines == []
    shape = photo_to_shape(w_line)
    assert len(shape) == 50
    assert shape.min() == 0 and shape.max() == 1
    assert len(find_local_minima(list(shape), 3)) == 2


# --- геометрия линий ------------------------------------------------------------------
def test_classify_parallel():
    a = Line(0, 100, 200, 50)
    b = Line(0, 200, 200, 150)
    assert "параллельны" in classify_pair(a, b)


def test_classify_diverging():
    a = Line(0, 100, 200, 50)
    b = Line(0, 120, 200, 200)
    assert "расходятся" in classify_pair(a, b)


# --- форма ----------------------------------------------------------------------------
def test_normalize():
    assert list(normalize([100, 200, 300])) == [0, 0.5, 1]
    assert list(normalize([5, 5, 5])) == [0, 0, 0]


def test_resample_keeps_ends():
    out = resample([0, 10], 5)
    assert len(out) == 5
    assert out[0] == 0 and out[-1] == 10
    assert np.allclose(out, [0, 2.5, 5, 7.5, 10])
