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
