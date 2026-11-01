"""Расписание автоскана и «присылать только новое»."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import autoscan
import storage
from autoscan import Schedule, is_new, next_run


def ts(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def test_next_run_after_candle_close():
    # 4h-свечи закрываются в 00, 04, 08 … UTC; проверяем через минуту после закрытия
    assert next_run("4h", ts("2026-11-01 09:30:00")) == ts("2026-11-01 12:01:00")
    assert next_run("1h", ts("2026-11-01 09:00:30")) == ts("2026-11-01 09:01:00")
    assert next_run("1d", ts("2026-11-01 09:30:00")) == ts("2026-11-02 00:01:00")


def test_weekly_candles_close_on_monday():
    # 2026-11-01 — воскресенье; недельная свеча Binance закрывается в понедельник 00:00 UTC
    assert next_run("1w", ts("2026-11-01 12:00:00")) == ts("2026-11-02 00:01:00")


def test_schedule_fires_once_per_close():
    schedule = Schedule(ts("2026-11-01 11:59:00"))
    assert "4h" not in schedule.due(ts("2026-11-01 12:00:30"))   # минута ещё не прошла
    assert "4h" in schedule.due(ts("2026-11-01 12:01:05"))
    assert "4h" not in schedule.due(ts("2026-11-01 12:05:00"))   # второй раз не срабатывает


def test_is_new_ignores_same_place():
    sent = [{"symbol": "BTCUSDT", "interval": "4h", "name": "Треугольник",
             "start": 1000, "end": 2000, "at": 0}]
    same = {"symbol": "BTCUSDT", "interval": "4h", "name": "Треугольник", "start": 1100, "end": 2100}
    other_place = {**same, "start": 5000, "end": 6000}
    other_coin = {**same, "symbol": "ETHUSDT"}
    assert not is_new(same, sent)
    assert is_new(other_place, sent)
    assert is_new(other_coin, sent)


def fake_df(n=50):
    times = pd.date_range("2026-10-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open_time": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})


def test_autoscan_sends_only_new(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    try:
        users = {"7": {"patterns": [{"id": "1", "name": "FVG", "photo": "x"}],
                       "intervals": ["4h"], "autoscan": True}}
        storage.save_users(users)
        df = fake_df()
        card = {"symbol": "BTCUSDT", "interval": "4h", "df": df, "history": False, "caption": "",
                "items": [{"name": "FVG", "score": 0.9, "start": 40, "window": 10,
                           "kind": "bullish_fvg", "zone": (99.0, 100.0)}]}
        autoscan.scan_user = lambda patterns, intervals, fetch: ([dict(card)], [])
        sent_cards, messages = [], []
        send_cards = lambda chat_id, cards: sent_cards.extend(cards)
        send_message = lambda text, chat_id: messages.append(text)

        assert autoscan.run_autoscan("4h", send_cards, send_message, now=1e9) == 1
        assert autoscan.run_autoscan("4h", send_cards, send_message, now=1e9 + 3600) == 0   # то же — молчим
        assert autoscan.run_autoscan("1d", send_cards, send_message, now=1e9) == 0          # 1d не выбран
        assert len(sent_cards) == 1 and "Автоскан 4H" in messages[0]
    finally:
        storage.FILE = original
