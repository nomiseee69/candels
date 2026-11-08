"""Настоящие скрины пользователей, на которых бот когда-то ошибался."""
import os

import numpy as np

import vision

HERE = os.path.join(os.path.dirname(__file__), "samples")


def test_fvg_box_touching_candle_and_candles_at_bottom():
    """Тёмная тема, без объёма. Рамка зоны стоит вплотную к большой красной свече, над рамкой —
    маленькая свеча; после падения свечи уходят в самый низ картинки.
    Раньше: рамку читал как зелёную свечу, свечи внизу терял (принимал за объём)."""
    candles, zones, _ = vision.extract_marks(os.path.join(HERE, "user_fvg_box_low.png"))
    assert candles is not None and len(candles) >= 24           # на скрине ~26 свечей
    assert zones and zones[0]["end"] > zones[0]["start"]
    start = zones[0]["start"]
    # у левого края зоны — настоящая большая красная свеча, а не зелёная «рамка»
    big = candles.iloc[start + 1]
    assert big["close"] < big["open"]
    assert (big["high"] - big["low"]) > 3 * np.median(candles["high"] - candles["low"])
    # после неё — свечи внизу, ниже зоны
    after = candles.iloc[start + 2:]
    assert len(after) >= 6 and (after["high"] < zones[0]["bottom"]).mean() > 0.7


def test_volume_still_detected_on_normal_screens(tmp_path):
    """Скрин с объёмом внизу: столбики объёма не становятся свечами."""
    import screen_exam
    df = screen_exam.market(1).iloc[100:170].reset_index(drop=True)
    path = screen_exam.render(df, str(tmp_path / "v.png"), "dark", volume=True)
    candles = vision.extract_candles(path)
    assert candles is not None and abs(len(candles) - len(df)) <= 3 and "volume" in candles


def test_no_volume_candles_reach_bottom(tmp_path):
    """Скрин без объёма: свечи у самого низа картинки не теряются."""
    import screen_exam
    df = screen_exam.market(2).iloc[300:380].reset_index(drop=True)
    path = screen_exam.render(df, str(tmp_path / "n.png"), "light", volume=False)
    candles = vision.extract_candles(path)
    assert candles is not None and abs(len(candles) - len(df)) <= 3


def _window(seed=1, a=100, b=170):
    import screen_exam
    return screen_exam.market(seed).iloc[a:b].reset_index(drop=True)


def test_huge_4k_screenshot_reads_like_normal(tmp_path):
    """Скрин с 4K-монитора: рамки и сглаживание толще — раньше остатки рамки читались сотнями «свечей»."""
    import screen_exam
    df = _window()
    path = screen_exam.render(df, str(tmp_path / "big.png"), "light", volume=True, dpi=280)
    candles = vision.extract_candles(path)
    assert candles is not None and abs(len(candles) - len(df)) <= 3


def test_thin_zone_is_found(tmp_path):
    """Тонкая зона (OB / FVG высотой в пару свечей) — тоже зона, которую нарисовал пользователь."""
    import screen_exam
    df = _window()
    span = df["high"].max() - df["low"].min()
    mid = float(df["close"].iloc[40])
    path = screen_exam.render(df, str(tmp_path / "thin.png"), "light", ("zone", (40, mid, mid + span * 0.03)),
                              volume=False)
    candles, zones, _ = vision.extract_marks(path)
    assert candles is not None and zones and abs(zones[0]["start"] - 40) <= 2


def test_red_lines_over_candles_are_drawings(tmp_path):
    """Линии красного цвета (как свечи) поверх графика — это рисунок пользователя, а не свечи."""
    import screen_exam
    df = _window(2, 300, 370)
    lo, hi = float(df["low"].iloc[30:45].min()), float(df["high"].iloc[30:45].max())
    saved = dict(screen_exam.DRAW)
    screen_exam.DRAW.update(line="#f23645")
    try:
        path = screen_exam.render(df, str(tmp_path / "red.png"), "light",
                                  ("lines", ((30, hi, 45, hi - (hi - lo) * 0.2), (30, lo, 45, lo + (hi - lo) * 0.1))),
                                  volume=False)
    finally:
        screen_exam.DRAW.clear()
        screen_exam.DRAW.update(saved)
    lines = vision.drawn_color_lines(vision.load_image(path))
    assert len(lines) >= 2 and all(getattr(ln, "candle_colored", False) for ln in lines)
    candles = vision.extract_candles(path)
    assert candles is not None and abs(len(candles) - len(df)) <= 3
