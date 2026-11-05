"""Оценки 👍/👎 в боте, несколько примеров на паттерн, «сохранить как пример»."""

import numpy as np
import pandas as pd

import scanner
import storage
from similarity import example_from_item
from test_trendlines import make_df, with_history, zigzag
from trendlines import CHANNELS, find_line_pattern


def frame(n=60, price=100.0):
    times = pd.date_range("2026-10-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open_time": times, "open": price, "high": price + 1,
                         "low": price - 1, "close": price})


def channel_df():
    df = make_df(with_history(zigzag(110, 95, 100, 85, n=80, waves=4)))
    df["open_time"] = pd.date_range("2026-09-01", periods=len(df), freq="4h", tz="UTC")
    return df


def test_feedback_storage_replaces_same_item(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    try:
        storage.add_feedback(1, {"key": "a", "name": "FVG", "vote": 1, "features": {}})
        storage.add_feedback(1, {"key": "a", "name": "FVG", "vote": -1, "features": {}})
        storage.add_feedback(1, {"key": "b", "name": "FVG", "vote": 1, "features": {}})
        assert [(r["key"], r["vote"]) for r in storage.get_feedback(1)] == [("a", -1), ("b", 1)]
        storage.clear_feedback(1)
        assert storage.get_feedback(1) == []
    finally:
        storage.FILE = original


def test_example_from_market_item():
    df = channel_df()
    found = find_line_pattern(df, CHANNELS)
    example = example_from_item(df, found)
    assert example["kind"] == "channel_down" and len(example["reference"]["geometry"]) == 4
    assert len(example["reference"]["path"]) == 50
    shape = example_from_item(df, {"name": "Двойное дно", "score": 0.9, "start": 10, "window": 30})
    assert len(shape["shape"]) == 50 and min(shape["shape"]) == 0 and max(shape["shape"]) == 1
    assert example_from_item(df, {"name": "FVG", "kind": "bullish_fvg", "zone": (1, 2),
                                  "start": 1, "window": 2}) is None   # SMC — по правилам


def test_market_example_used_as_reference():
    df = channel_df()
    found = find_line_pattern(df, CHANNELS)
    example = example_from_item(df, found)
    patterns = [{"name": "Канал", "photo": "нет_файла.png", "kind": "channel_down"},
                {"name": "Канал", "photo": "превью.png", "kind": "channel_down",
                 "reference": example["reference"]}]
    detectors, skipped = scanner.build_detectors(patterns)
    assert skipped == [] and len(detectors) == 1
    item = detectors[0][1](df)
    assert item is not None and item["similarity"] > 0.9


def test_many_shape_examples_are_one_detector():
    w = np.concatenate([np.linspace(1, 0, 10), np.linspace(0, 0.6, 10), np.linspace(0.6, 0, 10),
                        np.linspace(0, 1, 10)])
    flat = np.linspace(0, 1, 40)
    patterns = [{"name": "Двойное дно", "photo": "a", "shape": list(np.interp(np.linspace(0, 39, 50), range(40), flat))},
                {"name": "Двойное дно", "photo": "b", "shape": list(np.interp(np.linspace(0, 39, 50), range(40), w))}]
    detectors, _ = scanner.build_detectors(patterns)
    assert len(detectors) == 1                                    # раньше было бы два «Двойных дна»
    prices = list(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.2, 80))) + list(95 + 8 * w)
    item = detectors[0][1](make_df(prices))
    assert item is not None and item["example"] == 2              # похоже на второй пример
    assert scanner.describe_item({**item, "name": "Двойное дно"}) == "Двойное дно"   # без процентов


def test_judge_filters_scan():
    df_candles = [[int(t.timestamp() * 1000), 1, 2, 0.5, 1.5, 1] for t in
                  pd.date_range("2026-01-01", periods=150, freq="h", tz="UTC")]
    original = scanner.build_detectors
    scanner.build_detectors = lambda patterns, settings=None: (
        [("FVG", lambda df: {"score": 0.9, "start": 140, "window": 5})], [])
    try:
        fetch = lambda s, i, n: df_candles
        cards, _ = scanner.scan_user([{"name": "FVG"}], ["1h"], fetch, ["A"], judge=lambda n, d, it: (True, 0.8))
        assert cards[0]["items"][0]["fit"] == 0.8 and "%" not in cards[0]["caption"].split("http")[0]
        cards, _ = scanner.scan_user([{"name": "FVG"}], ["1h"], fetch, ["A"], judge=lambda n, d, it: (False, None))
        assert cards == []
    finally:
        scanner.build_detectors = original


def test_bot_votes_and_saves_example(tmp_path):
    import bot
    original = storage.FILE, bot.find_card, bot.send_message, bot.edit_keyboard, bot.PATTERNS_DIR
    storage.FILE = str(tmp_path / "users.json")
    messages, edits = [], []
    df = channel_df()
    found = {**find_line_pattern(df, CHANNELS), "name": "Канал"}
    fvg = {"name": "FVG", "score": 0.9, "start": len(df) - 5, "window": 5, "kind": "bullish_fvg",
           "zone": (90.0, 91.0), "distance": 0.01}
    one = {"symbol": "SOLUSDT", "history": False,
           "frames": [{"interval": "4h", "df": df, "items": [found], "history": False}]}
    two = {**one, "frames": [{"interval": "4h", "df": df, "items": [found, fvg], "history": False}]}
    cards = {1: one, 2: two}
    bot.find_card = lambda chat_id, card_id: cards[card_id]
    bot.send_message = lambda text, chat_id, keyboard=None: messages.append((text, keyboard))
    bot.edit_keyboard = lambda chat_id, message_id, keyboard: edits.append(keyboard)
    bot.PATTERNS_DIR = str(tmp_path)
    try:
        bot.vote_card(7, 1, 1)                                   # одна находка — сразу запомнили
        assert storage.get_feedback(7)[0]["vote"] == 1 and "запомнил" in messages[-1][0].lower()
        assert "как ещё один пример" in messages[-1][0] and messages[-1][1] is None   # 👍 сам стал примером
        assert storage.get_feedback(7)[0]["example"]["reference"]
        bot.vote_card(7, 2, -1)                                  # несколько — меню оценки
        rows = messages[-1][1]["inline_keyboard"]
        assert len(rows) == 3 and rows[0][-1]["text"] == "💾" and rows[1][-1]["text"] == "💾"   # FVG — отпечатком
        assert rows[-1][-1]["callback_data"] == "home"                     # выход в меню
        bot.rate_item(7, 2, 1, -1, 100)
        assert edits[-1]["inline_keyboard"][1][1]["text"] == "❌"
        bot.save_example(7, 1, 0)
        examples = [p for p in storage.get_patterns(7) if p["name"] == "Канал"]
        assert len(examples) == 1 and examples[0]["reference"] and examples[0]["note"].startswith("с рынка")
        assert "Сохранил как пример №1" in messages[-1][0]
    finally:
        storage.FILE, bot.find_card, bot.send_message, bot.edit_keyboard, bot.PATTERNS_DIR = original
