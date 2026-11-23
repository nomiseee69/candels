from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fingerprint
import scanner
import similarity
import smc
import trendlines
from screenshot_query import screenshot_choices
from vision import Line


def candles(n):
    return pd.DataFrame({"open": np.full(n, 100.), "high": np.full(n, 104.),
                         "low": np.full(n, 96.), "close": np.full(n, 101.)})


def test_real_photo_queries_keep_only_observed_evidence():
    samples = Path(__file__).with_name("samples")
    _, gap = next((label, ex) for label, ex in screenshot_choices(str(samples / "bearish_fvg.png"))
                  if ex["kind"] == "bearish_fvg")
    fp = gap["fingerprint"]
    assert fp["window"] is None and "significant" not in fp["features"]
    assert fp["features"]["gap_atr"] > 0 and fp["features"]["body_atr"] > 0
    _, triangle = next((label, ex) for label, ex in screenshot_choices(str(samples / "dark_triangle.png"))
                       if ex["kind"] == "symmetric")
    ref = triangle["reference"]
    upper_start, upper_end, lower_start, lower_end = ref["geometry"]
    assert ref["visible_span"] and ref["path"]
    assert upper_start - lower_start > upper_end - lower_end >= 0


@pytest.mark.parametrize("edge", ["high", "low"])
def test_censored_triplet_has_no_visual_score(edge):
    df = candles(50)
    df.attrs["price_pixel_bounds"] = (0, 199)
    item = {"start": 20, "end": 22, "impulse_and_wide": True}
    fp = {"features": {"significant": 0., "gap_atr": 2., "body_atr": 3.}, "window": fingerprint._window(df, item)}
    df.loc[22, edge] = 199 if edge == "high" else 0
    visible = fingerprint.visible_fvg_reference(fp, df, item)
    assert visible["window"] is None
    assert visible["features"] == {"gap_atr": 2., "body_atr": 3.}
    assert fp["window"] and "significant" in fp["features"]
    df.loc[21, "close"] = df.loc[21, edge] = df.loc[22, edge]
    assert fingerprint.visible_fvg_reference(fp, df, item) is None


def test_crop_only_trims_censored_context_and_keeps_known_weak_impulse():
    df = candles(50)
    df.attrs["price_pixel_bounds"] = (0, 199)
    df.loc[15, "low"] = 0
    df.loc[30, "high"] = 199
    item = {"start": 20, "end": 22, "impulse_and_wide": False}
    fp = {"features": {"significant": 0}, "window": fingerprint._window(df, item)}
    visible = fingerprint.visible_fvg_reference(fp, df, item)
    assert visible["window"]["ohlc"] == fp["window"]["ohlc"][16:30]
    assert (visible["window"]["start"], visible["window"]["end"]) == (4, 6)
    assert visible["features"]["significant"] == 0


def test_visible_failed_break_is_evidence_even_without_volume():
    df = candles(50)
    df.loc[10, "low"] = 90  # a confirmed swing low well before the bearish impulse
    item = {"start": 20, "end": 22, "impulse_and_wide": True, "kind": "bearish_fvg", "significant": False}
    fp = {"features": {"significant": 0}, "window": None}
    assert fingerprint.visible_fvg_reference(fp, df, item)["features"] == {"significant": 0}


def test_direct_line_reference_ignores_projection_beyond_last_candle():
    xs = np.arange(61)
    def analysis(end):
        return SimpleNamespace(ok=True, xs=xs, prices=np.sin(xs), height=120,
                               lines=[Line(0, 5, end, 5 + end * .5), Line(0, 100, end, 100 - end * .5)])
    near = similarity.reference_from_analysis(analysis(60), visible_span=True)
    far = similarity.reference_from_analysis(analysis(110), visible_span=True)
    np.testing.assert_allclose(near["geometry"], far["geometry"])
    assert far["geometry"][1] >= far["geometry"][3]
    assert similarity.reference_to_json(far)["visible_span"] is True
    assert "visible_span" not in similarity.reference_from_analysis(analysis(110))


def test_saved_and_photo_line_references_can_be_compared_together():
    df = candles(50)
    df["close"] = 100 + np.sin(np.arange(len(df)))
    item = {"upper": (2, 106., 42, 102.), "lower": (12, 95., 48, 98.)}
    saved = similarity.figure_shape(df, item)
    photo = {**similarity.figure_shape(df, item, visible_span=True), "visible_span": True}
    for index, ref in enumerate((saved, photo)):
        refs = iter([None] * index + [ref])
        score, best = similarity.best_match(df, item, refs)
        assert score == pytest.approx(1) and best == index
    score, _ = similarity.best_match(df, item, iter([saved, photo]))
    assert score == pytest.approx(1)


def test_photo_gap_selects_among_admissible_candidates(monkeypatch):
    df = candles(50)
    def gap(i, kind, sim):
        return {"start": i, "kind": kind, "score": .90, "distance": 0, "test_similarity": sim}
    opposite = gap(40, "bullish_fvg", .99)
    unlike = gap(41, "bearish_fvg", .59)
    match = gap(42, "bearish_fvg", .75)
    pool = [opposite, unlike, match]
    monkeypatch.setattr(smc, "find_all_fvg", lambda *args, **kwargs: pool)
    monkeypatch.setattr(scanner, "best_fingerprint", lambda fps, df, it: (it["test_similarity"], 0))
    monkeypatch.setattr(scanner, "rate", lambda df, it: it)
    settings = {"quality": "off", "similar_only": True, "photo_kind": "bearish_fvg"}
    assert scanner.find_rule(df, "FVG", settings, [{}]) is match
    pool.remove(match)
    assert scanner.find_rule(df, "FVG", settings, [{}]) is None
    assert scanner.find_rule(df, "FVG", {"quality": "off"}, [{}]) is opposite


def test_line_window_must_pass_before_ranking(monkeypatch):
    df = candles(125)
    def fit(highs, lows, closes):
        n = len(highs)
        if n not in (30, 50, 120):
            return None
        return {"start": 0, "window": n, "score": .82 if n == 30 else 1.,
                "kind": "ascending" if n == 50 else "symmetric", "touch_points": [],
                "upper": (0, 104., n-1, 102.), "lower": (0, 96., n-1, 98.)}
    monkeypatch.setattr(trendlines, "_analyze_window", fit)
    monkeypatch.setattr(scanner, "best_match", lambda df, it, refs: ({30:.61, 50:.99, 120:.59}[it["window"]], 0))
    result = scanner.find_lines(df, "Треугольник", "symmetric", [{}], similar_only=True, photo_kind="symmetric")
    assert (result["kind"], result["window"]) == ("symmetric", 30)
