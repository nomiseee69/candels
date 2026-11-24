import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import fingerprint
import similarity
import trendlines
import vision


SAMPLES = Path(__file__).with_name("samples")


def test_annotation_plate_clears_its_text_and_leaves_price_strokes():
    background = np.array([10, 10, 10], dtype=np.uint8)
    fill = np.array([33, 33, 33], dtype=np.uint8)
    image = np.empty((44, 230, 3), dtype=np.uint8)
    image[:] = fill
    # Узкая цветная тень и штрих монохромной свечи рядом с подписью.
    image[2:40, 5] = (80, 180, 30)
    image[2:40, 223] = (195, 195, 195)
    box = (0, 0, 229, 43, fill)
    assert vision._text_plate_ink(image, box) is None
    cv2.putText(image, "LEVEL BROKEN", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (215, 215, 215), 2)
    ink = vision._text_plate_ink(image, box)
    assert ink is not None and ink.any()
    cleaned = vision._erase_text_plates(image, [(box, ink)], background)
    assert np.all(cleaned[ink] == background) and np.all(cleaned[10, 10] == background)
    np.testing.assert_array_equal(cleaned[:, (5, 223)][2:40], image[:, (5, 223)][2:40])
    assert vision._inside_plate(vision.Line(0, 0, 229, 0), box[:4], (0, 0))
    assert not vision._inside_plate(vision.Line(0, 24, 240, 24), box[:4], (0, 0))


def test_wedge_reference_keeps_both_boundaries():
    path = str(SAMPLES / "falling_wedge.png")
    analysis = vision.analyze_screenshot(path)
    candles, zones, lines = vision.extract_marks(path, include_x=True)
    assert trendlines.kind_from_drawing(analysis.lines) == "falling_wedge"
    assert 140 <= len(candles) <= 146 and len(lines) == 2 and not zones
    assert candles.x.iloc[0] < 10 and candles.x.iloc[-1] > 690
    assert len(similarity.reference_from_screenshot(path)["path"]) == 50


def test_recognized_boundary_pair_is_not_reused_as_a_neckline(monkeypatch):
    import screenshot_query

    def unexpected_reversal(*args):
        pytest.fail("Обе линии уже связаны с границами треугольника")

    monkeypatch.setattr(screenshot_query, "_marked_reversals", unexpected_reversal)
    [(_, choice)] = screenshot_query.screenshot_choices(SAMPLES / "dark_triangle.png")
    assert choice["kind"] == "symmetric" and choice["reference"]["geometry"]


def test_highlighted_bearish_gap():
    candles, zones, lines = vision.extract_marks(str(SAMPLES / "bearish_fvg.png"))
    assert 41 <= len(candles) <= 45 and len(zones) == 1 and not lines
    fp, item = fingerprint.screenshot_fingerprint("FVG", candles, zones=zones)
    assert item["kind"] == "bearish_fvg" and fp["family"] == "fvg"
    assert not item.get("from_zone") and abs(item["start"] - zones[0]["start"]) <= 1
    np.testing.assert_allclose(item["zone"], [zones[0]["bottom"], zones[0]["top"]], atol=6)


def test_price_does_not_follow_volume_before_ambiguity_check():
    img, h, _, _, mask = vision._prepare(str(SAMPLES / "cluttered_structure.png"))
    rows = vision._candle_rows(img, vision._colored(img, mask), h, vision._zone(mask, h), full=mask)
    tail = vision._to_frame(rows, h).tail(8)
    assert 815 <= tail.x.iloc[-1] <= 840
    assert (h - 1 - tail.low < 650).all()
    assert 550 < h - 1 - tail.close.iloc[-1] < 580


@pytest.mark.parametrize("name", ["Клин", "FVG", "Range", "Двойное дно"])
def test_cluttered_upload_requests_one_target_without_saving(tmp_path, monkeypatch, name):
    import bot
    import storage

    messages = []
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr(bot, "PATTERNS_DIR", str(tmp_path))
    monkeypatch.setattr(bot, "pending_spec", {})
    monkeypatch.setattr(bot, "download_photo", lambda _, path: shutil.copy(SAMPLES / "cluttered_structure.png", path))
    monkeypatch.setattr(bot, "send_message", lambda text, chat_id, kb: messages.append((text, kb)))
    monkeypatch.setattr(bot, "send_photo", lambda *args: pytest.fail("Ненадёжный эталон принят"))
    bot.receive_pattern_photo(1, name, "local-test-image")
    assert "Оставь один" in messages[-1][0]
    assert messages[-1][1] == bot.retry_keyboard(bot.ALL_PATTERNS.index(name))
    assert not storage.get_patterns(1) and not list(tmp_path.glob("*.jpg"))


def test_price_scale_labels_cannot_collapse_onto_last_candle():
    df = pd.DataFrame({"x": [10, 20, 30, 40], "low": 55, "high": 70})
    boxes = [(50, 10, 60, 20, None), (20, 30, 60, 40, None), (12, 1, 38, 8, None)]
    zones = vision._zones_on_candles(df, boxes, 100)
    assert [(z["start"], z["end"]) for z in zones] == [(1, 3)]
    lines = [vision.Line(50, y, 60, y) for y in (10, 20)]
    assert not vision._lines_on_candles(df, lines, 100)
    legend = [vision.Line(12, y, 38, y) for y in (1, 8)]
    assert not vision._zones_from_lines(vision._lines_on_candles(df, legend, 100), 100, df)


def test_volume_baseline_requires_separate_bars():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[150:175, 25:295] = (120, 160, 20)  # цветная подпись или зона
    assert vision._colored_volume_base(image) is None
    for j, x in enumerate(range(15, 285, 6)):
        image[207 - j % 13:224, x:x + 2] = (80, 145, 25)
    assert vision._colored_volume_base(image) == 223


@pytest.mark.parametrize("filename", ["real_tv_light_table.png", "dark_avax_labels.png"])
def test_annotated_examples_still_have_a_price_series(filename):
    analysis = vision.analyze_screenshot(str(SAMPLES / filename))
    assert analysis.ok and analysis.mode == "candles"
    assert np.isfinite(analysis.prices).all()
