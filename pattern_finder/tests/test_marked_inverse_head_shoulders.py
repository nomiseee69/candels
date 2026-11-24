import numpy as np
import pytest

import screenshot_query
from shapes_check import _shoulder_geometry, marked_inverse_head_shoulders, validate
from test_shapes_check import W, frame
from vision import UnreliableScreenshot

NAME = "Перевёрнутая ГиП"


def recovered_candles(extra_swing=False):
    points = [(0, 90), (12, 105), (20, 97), (37, 117), (45, 98), (52, 105), (64, 91)]
    if extra_swing:
        points[3:3] = [(27, 110), (30, 106)]
    return frame(points, mirror=True).iloc[60:].reset_index(drop=True)


def neckline_mark(df):
    first = int(df.high.iloc[18:23].idxmax())
    second = int(df.high.iloc[42:48].idxmax())
    y0 = float(df.high.iloc[first])
    slope = (float(df.high.iloc[second]) - y0) / (second - first)
    return {"x0": first, "x1": 62, "y0": y0, "y1": y0 + slope * (62 - first), "flat": False}


@pytest.mark.parametrize("zones", [[], [{"start": 54, "end": 56, "top": 112, "bottom": 110}]])
def test_single_neckline_does_not_disable_inverse_head_shoulders(monkeypatch, zones):
    df = recovered_candles()
    assert validate(NAME, df, 0, len(df))["confirmed"]
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, zones, [neckline_mark(df)]))
    assert [item["name"] for _, item in screenshot_query.screenshot_choices("decoded")] == [NAME]


def test_mark_can_disambiguate_shoulder_from_swing_on_the_head_slope():
    df = recovered_candles(extra_swing=True)
    diagnostics = {}
    assert validate(NAME, df, 0, len(df), diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "shoulder_neck_clearance"
    assert diagnostics["shoulders"][0] == 28
    assert diagnostics["value"] < diagnostics["limit"]

    result, _ = marked_inverse_head_shoulders(df, [neckline_mark(df)])
    assert result["confirmed"]
    assert [x for x, _ in result["points"][:5]] == [13, 20, 38, 45, 53]


def test_bearish_head_shoulders_uses_confirmed_neckline_and_individual_contour(monkeypatch):
    from scanner import find_photo_shape

    inverse = recovered_candles()
    mark = neckline_mark(inverse)
    df = inverse.assign(open=240 - inverse.open, close=240 - inverse.close,
                        high=240 - inverse.low, low=240 - inverse.high)
    line = {**mark, "y0": 240 - mark["y0"], "y1": 240 - mark["y1"]}
    noise = {**line, "y0": line["y0"] + 20, "y1": line["y1"] + 20}
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], [line, noise, noise]))
    [(label, reference)] = screenshot_query.screenshot_choices("decoded")
    assert label == reference["name"] == "Голова и плечи"
    assert find_photo_shape(df, reference)["confirmed"]

    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], [noise]))
    with pytest.raises(UnreliableScreenshot):
        screenshot_query.screenshot_choices("decoded")


def test_one_bar_drop_from_head_needs_independent_neckline():
    highs = np.full(18, 12.)
    lows = np.full(18, 10.)
    closes = np.full(18, 11.)
    highs[[3, 8, 12]] = [15, 22, 16]
    lows[[6, 9, 15]] = [7, 8, 5]
    closes[15] = 6
    diag = {}
    args = (highs, lows, closes, 2, 3, 8, 12)
    assert _shoulder_geometry(*args, lambda **details: diag.update(details)) is None
    assert diag["reason"] == "compressed_landmarks"
    marked = _shoulder_geometry(*args, lambda **_: None, head_gap=1)
    assert marked["confirmed"] and [x for x, _ in marked["points"][:5]] == [3, 6, 8, 9, 12]


@pytest.mark.parametrize("case", ["wrong_neckline", "wick_only", "double_bottom", "sideways"])
def test_neckline_mark_does_not_replace_confirmation(monkeypatch, case):
    df = recovered_candles()
    mark = neckline_mark(df)
    if case == "wrong_neckline":
        mark["y1"] += 12
    elif case == "wick_only":
        cols = ["open", "low", "close"]
        df.loc[54:, cols] = df.loc[54:, cols].clip(upper=139)
    elif case == "double_bottom":
        df = frame(W).iloc[60:].reset_index(drop=True)
    else:
        df = frame([(0, 139), (16, 140), (32, 139), (48, 140), (64, 139)]).iloc[60:].reset_index(drop=True)
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], [mark]))
    with pytest.raises(UnreliableScreenshot):
        screenshot_query.screenshot_choices("decoded")

def test_photo_scanner_transfers_marked_shoulders_to_another_price_scale(monkeypatch):
    from scanner import find_photo_shape

    df = recovered_candles(extra_swing=True)
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], [neckline_mark(df)]))
    reference = dict(screenshot_query.screenshot_choices("decoded"))[NAME]
    foreign = df.copy()
    prices = ["open", "high", "low", "close"]
    foreign[prices] = foreign[prices] * 7 + 130
    found = find_photo_shape(foreign, reference)
    assert found["confirmed"] and found["score"] == pytest.approx(1)
    assert find_photo_shape(foreign, {**reference, "name": "Голова и плечи"}) is None

def test_short_marked_photo_can_reach_result_card(monkeypatch):
    from scanner import scan_user

    df = recovered_candles(extra_swing=True)
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], [neckline_mark(df)]))
    reference = dict(screenshot_query.screenshot_choices("decoded"))[NAME]
    candles = [[i * 3600000, row.open, row.high, row.low, row.close, 0]
               for i, row in enumerate(df.itertuples())]
    cards, skipped = scan_user([reference], ["1h"], fetch=lambda *_: candles, symbols=["TEST"],
                               settings={"similar_only": True, "quality": "off"})
    assert not skipped
    assert cards[0]["items"][0]["score"] == pytest.approx(1)
