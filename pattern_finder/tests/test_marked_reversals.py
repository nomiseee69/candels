import pytest

from scanner import find_photo_shape
import screenshot_query
from shapes_check import validate
from test_shapes_check import W, frame
from vision import UnreliableScreenshot


@pytest.mark.parametrize("name,mirror", [("Двойное дно", False), ("Двойная вершина", True)])
def test_marked_double_reversal(monkeypatch, name, mirror):
    df = frame(W, mirror=mirror)
    _, left, neck, right, _ = validate(name, df, 60, 61)["points"]
    marks = [{"x0": x, "x1": xx, "y0": y, "y1": yy, "flat": False}
             for (x, y), (xx, yy) in ((left, neck), (neck, right))]
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (df, [], marks))
    monkeypatch.setattr(screenshot_query, "drawn_span_has_swings", lambda *_: False)
    [(label, item)] = screenshot_query.screenshot_choices("decoded")
    assert (label, item["name"]) == (name, name)
    assert find_photo_shape(df, item)["confirmed"]

    # Та же разметка над боковиком не заменяет подтверждение по свечам.
    sideways = frame([(0, 100.3), (15, 100), (30, 100.4), (45, 100.1), (60, 100.4)])
    monkeypatch.setattr(screenshot_query, "extract_marks", lambda _: (sideways, [], marks))
    with pytest.raises(UnreliableScreenshot):
        screenshot_query.screenshot_choices("decoded")
