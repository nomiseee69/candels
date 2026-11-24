import pytest

import screenshot_query
from shapes_check import marked_inverse_head_shoulders, validate
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
