"""Двойное дно / вершина, голова и плечи: после похожести формы — проверка по точкам."""

import numpy as np
import pandas as pd
import pytest

from history import history_shapes
from scanner import WINDOWS, find_shape
from shapes_check import validate
from templates import template_shape


@pytest.mark.parametrize("name,mirror", [("Голова и плечи",False), ("Перевёрнутая ГиП",True)])
def test_photo_compares_landmarks_after_structural_validation(name, mirror):
    from scanner import build_detectors
    from shape import normalize, resample
    from shapes_check import landmark_closes
    points = [(0,95), (11,105), (20,100), (31,113), (41,100), (51,105), (66,96)]
    df = frame(points, mirror=mirror)
    figure = validate(name, df, 60, 67)
    assert figure and figure["points"][0][0] > 60
    shape = normalize(resample(landmark_closes(df,figure),50)).tolist()
    query = {"name":name,"shape":shape,"shape_window":67}
    jobs, _ = build_detectors([query], {"similar_only":True})
    result = jobs[0][1](df)
    assert result["score"] > .99 and result["points"][2][0] == figure["points"][2][0]
    straight = frame([(0,80),(66,120)])
    assert jobs[0][1](straight) is None


def frame(points, mirror=False, seed=0, before=60):
    """Случайный подход + ломаная points [(свеча, цена)] → свечи."""
    rng = np.random.default_rng(seed)
    xs, ys = zip(*points)
    body = np.interp(np.arange(xs[-1] + 1), xs, ys) + rng.normal(0, 0.25, xs[-1] + 1)
    lead = np.array(ys[0] + 6 + np.cumsum(rng.normal(-0.1, 0.3, before)))
    lead = lead - lead[-1] + ys[0]
    closes = np.concatenate([lead, body])
    if mirror:
        closes = 240 - closes
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC"),
                         "open": opens, "high": np.maximum(opens, closes) + 0.4,
                         "low": np.minimum(opens, closes) - 0.4, "close": closes})


W = [(0, 121), (15, 100), (30, 112), (45, 100.6), (60, 114)]          # дно, шея 112, дно, отскок


def test_double_bottom_and_top_found_with_points():
    df = frame(W)
    found = find_shape(df, [np.array(template_shape("Двойное дно"))], [True], "Двойное дно")
    assert found is not None and "шея" in found["label"]
    lows = [y for _, y in found["points"][1:4:2]]
    assert abs(lows[0] - lows[1]) < 2 and found["neckline"][1] > max(lows) + 5
    top = find_shape(frame(W, mirror=True), [np.array(template_shape("Двойная вершина"))], [True], "Двойная вершина")
    assert top is not None and top["neckline"][1] < min(y for _, y in top["points"][1:4:2])


def test_uneven_bottoms_are_not_a_double_bottom():
    uneven = [(0, 121), (15, 100), (30, 112), (45, 107), (60, 114)]     # второе дно сильно выше
    df = frame(uneven)
    assert validate("Двойное дно", df, 60, 61) is None
    flat = frame([(0, 100.3), (15, 100), (30, 100.4), (45, 100.1), (60, 100.4)])   # мелкая пила
    assert validate("Двойное дно", flat, 60, 61) is None


def test_history_keeps_only_real_double_bottoms():
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.standard_t(3, 1000) * 0.01))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    noise = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.003,
                          "low": np.minimum(opens, closes) * 0.997, "close": closes})
    shape = np.array(template_shape("Двойное дно"))
    raw = history_shapes(noise, shape, WINDOWS, 0.85)
    checked = history_shapes(noise, shape, WINDOWS, 0.85, "Двойное дно")
    assert len(checked) <= len(raw) and all("neckline" in f for f in checked)


def test_head_and_shoulders_rules():
    hs = [(0, 100), (10, 108), (18, 103), (30, 115), (42, 103.3), (52, 108.5), (60, 101)]
    found = validate("Голова и плечи", frame(hs), 60, 61)
    assert found is not None and len(found["points"]) == 6 and found["confirmed"]
    no_head = [(0, 100), (10, 108), (18, 103), (30, 108.3), (42, 103.3), (52, 108.5), (60, 101)]
    assert validate("Голова и плечи", frame(no_head), 60, 61) is None


def test_double_bottom_drawn_by_points_on_tradingview_and_picture(tmp_path):
    from chart import plot_example, plot_findings
    from pine import pine_script
    df = frame(W)
    found = {**find_shape(df, [np.array(template_shape("Двойное дно"))], [True], "Двойное дно"),
             "name": "Двойное дно"}
    code = pine_script("X", "1h", [(df, [found], False)])
    assert code.count("line.new(") == len(found["points"]) - 1 + 1        # ломаная + шея
    assert "line.style_dashed" in code and '"Двойное дно"' in code and "box.new" not in code
    plot_findings(df, "X", [found], str(tmp_path / "a.png"))
    plot_example(df, "X", "1h", found, str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()
