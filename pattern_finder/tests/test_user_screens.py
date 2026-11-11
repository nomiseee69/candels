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


def _sane(candles):
    """Свечи похожи на график: нет «свечей» на всю высоту из подписей и ярлыков."""
    span = candles["high"].max() - candles["low"].min()
    tall = ((candles["high"] - candles["low"]) > 0.4 * span).mean()
    return tall <= 0.02


def test_real_tradingview_full_window_light():
    """Настоящий скрин окна TradingView: подписи сверху, шкала цены, ярлыки, табличка.
    Раньше подписи и ярлыки читались как сотни свечей на всю высоту, а плотные свечи слипались.
    Настоящее число свечей посчитано по шкале времени: 1H за 9 дней ≈ 220, 1D за 5 месяцев ≈ 153."""
    for name, low, high in (("real_tv_light_full.png", 200, 240), ("real_tv_light_table.png", 143, 165),
                            ("real_tv_dark_labels.png", 300, 680)):
        candles = vision.extract_candles(os.path.join(HERE, name))
        assert candles is not None and low <= len(candles) <= high, (name, None if candles is None else len(candles))
        assert _sane(candles), name


def test_real_screenshot_keeps_the_shape_of_the_chart():
    """Дневной график BTC: полгода внизу, резкий рост, потом наверху. Таблица индикатора в углу,
    ярлык цены и пунктир последней цены не должны попасть в свечи, а свечи роста — потеряться."""
    candles = vision.extract_candles(os.path.join(HERE, "real_tv_light_table.png"))
    middle = ((candles["high"] + candles["low"]) / 2).to_numpy()
    span = candles["high"].max() - candles["low"].min()
    n = len(middle)
    assert middle[int(0.6 * n):].mean() - middle[:int(0.4 * n)].mean() > 0.3 * span     # справа цена выше
    steps = abs(middle[1:] - middle[:-1])
    assert (steps > 0.25 * span).sum() == 0            # нет «прыжков» на таблицу и обратно
    last = candles.tail(8)
    assert (last["high"] - last["low"]).max() > 3      # в конце настоящие свечи, а не точки пунктира


def test_zone_side_next_to_candles_is_not_a_wick():
    """Скрин пользователя: левая сторона зоны вплотную к маленькой свече сверху и к большой красной
    свече справа. Раньше сторона рамки приклеивалась к маленькой свече — у неё появлялась длинная
    «тень» на всю высоту зоны, а у красной свечи срезалось тело."""
    candles, zones, _ = vision.extract_marks(os.path.join(HERE, "user_fvg_box_side.png"))
    assert candles is not None and zones
    zone = zones[0]
    height = zone["top"] - zone["bottom"]
    first, big = candles.iloc[zone["start"]], candles.iloc[zone["start"] + 1]
    assert first["high"] - first["low"] < 0.35 * height                  # у маленькой свечи нет «тени» рамки
    assert big["close"] < big["open"] and big["open"] > zone["top"] - 0.2 * height   # тело красной — от верха зоны
