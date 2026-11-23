import numpy as np
import pytest

from fingerprint import marked_items
from flags import find_flag, flag_at_lines
from ranges import find_range, ranges_at_box
from test_ranges_flags import flag_df, frame, range_df
from trendlines import drawn_span_has_swings


def as_mark(line):
    x, y, xx, yy = line
    return {"x0": x, "y0": y, "x1": xx, "y1": yy, "flat": y == yy}


@pytest.mark.parametrize("shape,name", [("flag","Флаг"), ("pennant","Вымпел")])
@pytest.mark.parametrize("up", [True, False])
def test_body_on_screenshot_with_duplicate_edge(shape, name, up):
    df = flag_df(shape, up)
    target = find_flag(df, shape)
    top, bottom = [as_mark(target[k]) for k in ("upper","lower")]
    score, found = next(marked_items(name, df, [], [top, bottom, top.copy()]))
    assert score >= .70 and found["kind"] == target["kind"]
    assert flag_at_lines(df, shape, top, top.copy()) is None


def test_box_still_requires_oscillations():
    df = range_df()
    target = find_range(df)
    candidates = list(ranges_at_box(df,target["start"],target["end"]))
    assert any(c["start"] == target["start"] and c["end"] == target["end"] for c in candidates)
    straight = frame(np.linspace(90,105,len(df)))
    assert not list(ranges_at_box(straight,target["start"],target["end"]))


def test_drawn_boundaries_require_swings_in_the_annotated_interval():
    df = frame(np.r_[np.sin(np.arange(50)), np.arange(50)])
    assert drawn_span_has_swings(df,[{"x0":2,"x1":48,"y0":0,"y1":1}]*2)
    assert not drawn_span_has_swings(df,[{"x0":2,"x1":48,"y0":5,"y1":6}]*2)
    assert not drawn_span_has_swings(df,[{"x0":54,"x1":98,"y0":10,"y1":20}]*2)
