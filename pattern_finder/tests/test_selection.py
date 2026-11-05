"""Кнопка «🧩 Паттерны»: пользователь выбирает, что нанести на график."""

import pandas as pd

import scanner
from pine import pine_help, pine_script


def frame_df(n=60, freq="4h"):
    times = pd.date_range("2026-10-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})


FVG = {"name": "FVG", "score": 0.95, "start": 50, "window": 10, "kind": "bullish_fvg",
       "zone": (99.0, 100.0), "filled_at": None}
TRI = {"name": "Треугольник", "score": 0.93, "start": 10, "window": 50, "touches": 6,
       "upper": (10, 104.0, 59, 101.0), "lower": (12, 96.0, 59, 99.0), "touch_points": []}


def coin():
    frames = [
        {"symbol": "SOLUSDT", "interval": "1h", "df": frame_df(freq="h"), "items": [FVG], "history": False},
        {"symbol": "SOLUSDT", "interval": "4h", "df": frame_df(), "items": [TRI], "history": False},
    ]
    return {"symbol": "SOLUSDT", "history": False, "frames": frames}


def fake_candles(calls):
    def fetch(symbol, interval, limit):
        calls.append(interval)
        return [[int(t.timestamp() * 1000), 1, 2, 0.5, 1.5, 1] for t in
                pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")]
    return fetch


def test_only_chosen_patterns_and_timeframes():
    calls = []
    original = scanner.get_candles
    scanner.get_candles = fake_candles(calls)
    try:
        c = coin()
        groups = scanner.selected_markup([{"name": "FVG"}, {"name": "Треугольник"}], c,
                                         [("4h", c["frames"][1]["items"][0])])
    finally:
        scanner.get_candles = original
    assert calls == ["4h"]                                  # 1H не выбран — его даже не качаем
    assert {g[3] for g in groups} == {"4h"}
    current = [it for _, items, history, _ in groups if not history for it in items]
    assert [it["name"] for it in current] == ["Треугольник"]
    past = [it for _, items, history, _ in groups if history for it in items]
    assert all(it["name"] == "Треугольник" for it in past)  # история — только того же паттерна


def test_history_switch_in_script_and_help():
    groups = [(frame_df(), [TRI], False, "4h")]
    on = pine_script("SOLUSDT", "4h", groups, history_default=True)
    off = pine_script("SOLUSDT", "4h", groups, history_default=False)
    assert 'show_history = input.bool(true' in on and 'show_history = input.bool(false' in off
    text = pine_help("SOLUSDT", "4h", groups, selected=True)
    assert text.startswith("Разметка SOLUSDT\nТолько выбранное: Треугольник.")


def test_bot_toggles_selection():
    import bot
    edited = []
    original = bot.find_card, bot.edit_keyboard
    c = coin()
    bot.find_card = lambda chat_id, card_id: c
    bot.edit_keyboard = lambda chat_id, message_id, keyboard: edited.append(keyboard)
    try:
        bot.selections.clear()
        rows = bot.patterns_keyboard(1, 9, c)["inline_keyboard"]
        assert rows[0][0]["text"].startswith("✅ 1H · FVG")
        assert rows[1][0]["text"] == "✅ 4H · Треугольник"
        bot.change_selection(1, 9, 100, "sel", 0)            # снять FVG
        assert edited[-1]["inline_keyboard"][0][0]["text"].startswith("▫️ 1H · FVG")
        bot.change_selection(1, 9, 100, "shist")
        assert "Прошлые случаи этих паттернов: нет" in edited[-1]["inline_keyboard"][3][0]["text"]   # по умолчанию «да»
        bot.change_selection(1, 9, 100, "snone")
        assert bot.selection(1, 9)["off"] == {0, 1}
        bot.change_selection(1, 9, 100, "sall")
        assert bot.selection(1, 9)["off"] == set()
        assert bot.card_keyboard(9, c)["inline_keyboard"][0][0]["callback_data"] == "pats:9"
    finally:
        bot.find_card, bot.edit_keyboard = original
