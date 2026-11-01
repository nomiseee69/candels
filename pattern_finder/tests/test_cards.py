"""Таймфреймы, отчёт по монетам, разметка Pine, разбиение длинных сообщений."""

import pandas as pd

import storage
from history import coin_report
from pine import pine_script


def df_with_times(n=50, price=100.0):
    times = pd.date_range("2026-10-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": price, "high": price + 1, "low": price - 1, "close": price,
    })


# --- таймфреймы пользователя ------------------------------------------------------------------
def test_intervals_toggle(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")      # временный файл, настоящий users.json не трогаем
    try:
        allowed = ["1h", "4h", "12h", "1d", "1w"]
        assert storage.get_intervals(1) == ["4h"]                       # по умолчанию
        assert storage.toggle_interval(1, "1d", allowed) == ["4h", "1d"]
        assert storage.toggle_interval(1, "1h", allowed) == ["1h", "4h", "1d"]   # порядок как в меню
        assert storage.toggle_interval(1, "4h", allowed) == ["1h", "1d"]
        storage.toggle_interval(1, "1h", allowed)
        assert storage.toggle_interval(1, "1d", allowed) == ["1d"]      # последний не выключается
    finally:
        storage.FILE = original


# --- разметка для TradingView ---------------------------------------------------------------------
def test_pine_current_triangle_lines_end_at_apex():
    df = df_with_times()
    # верх 110 → 104, низ 90 → 98 на 39 свечах: сходятся, вершина впереди
    items = [{"name": "Треугольник", "score": 0.97, "start": 10, "window": 40,
              "upper": (10, 110.0, 49, 104.0), "lower": (10, 90.0, 49, 98.0)}]
    code = pine_script("BTCUSDT", "4h", [(df, items, False)])
    t10 = int(df["open_time"].iloc[10].timestamp() * 1000)
    assert code.startswith("//@version=5")
    assert 'indicator("Pattern Finder · BTCUSDT 4h", overlay=true' in code
    assert f"line.new({t10}, 110," in code
    assert code.count("line.new(") == 2
    assert "extend.right" not in code                 # линии не уходят в бесконечность
    # вершина: 104 - 6/39*k = 98 + 8/39*k → k ≈ 16.7 свечи после конца, цена 104 - 2.57 = 101.43
    assert code.count("101.4285714286") == 2         # обе линии сходятся в одной точке
    assert '"Треугольник 97%"' in code and "color.blue" in code


def test_pine_open_and_closed_fvg():
    df = df_with_times()
    items = [
        {"name": "FVG", "score": 0.9, "start": 20, "window": 30, "kind": "bullish_fvg",
         "zone": (100.5, 102.0), "filled_at": None},
        {"name": "FVG", "score": 0.9, "start": 5, "window": 45, "kind": "bearish_fvg",
         "zone": (95.0, 96.0), "filled_at": 12},
    ]
    code = pine_script("BTCUSDT", "4h", [(df, items, True)])
    t12 = int(df["open_time"].iloc[12].timestamp() * 1000)
    assert code.count("box.new(") == 2
    assert 'extend=extend.right' in code and 'text="FVG"' in code      # открытый — вправо
    assert f"{t12}, 95, xloc=xloc.bar_time, extend=extend.none" in code # закрытый — до свечи 12


def test_pine_history_shape_colored_by_outcome():
    df = df_with_times()
    df.loc[20, "high"] = 130
    df.loc[25, "low"] = 70
    items = [{"name": "Двойное дно", "score": 0.9, "start": 15, "window": 20,
              "outcome": {"change": 0.03}}]
    code = pine_script("ETHUSDT", "1d", [(df, items, True)])
    assert ", 130, " in code and ", 70, " in code
    assert "color.green" in code


def test_pine_small_prices_without_exponent():
    df = df_with_times(price=0.00001234)
    items = [{"name": "FVG", "score": 0.9, "start": 5, "window": 3, "kind": "bearish_fvg",
              "zone": (0.0000123, 0.00001245)}]
    code = pine_script("PEPEUSDT", "1h", [(df, items, False)])
    assert "e-0" not in code
    assert "0.0000123" in code


def test_pine_respects_object_limit():
    df = df_with_times(600)
    items = [{"name": "FVG", "score": 0.9, "start": i, "window": 2, "kind": "bullish_fvg",
              "zone": (100.0, 101.0), "filled_at": i + 1} for i in range(550)]
    code = pine_script("BTCUSDT", "1h", [(df, items, True)])
    assert code.count("box.new(") == 450


# --- отчёт по монете ---------------------------------------------------------------------------
def test_coin_report_lists_dates_and_results():
    df = df_with_times(100)
    found_shapes = [
        {"start": 10, "window": 20, "score": 0.9, "outcome": {"change": 0.025}},
        {"start": 60, "window": 20, "score": 0.9, "outcome": None},
    ]
    found_lines = [{"start": 30, "window": 20, "score": 0.9, "kind": "symmetric",
                    "breakout": "down", "outcome": {"change": -0.04}}]
    text = coin_report("BTCUSDT", df, [
        ("Двойное дно", "shape", found_shapes),
        ("Треугольник", "lines", found_lines),
        ("FVG", "fvg", []),
    ])
    assert text.startswith("🪙 BTCUSDT")
    assert "• Двойное дно ×2:" in text
    assert "рано судить" in text and "↑ +2.5%" in text
    assert "симметричный, пробой вниз (↓ -4.0%)" in text


def test_false_breakout_is_marked():
    df = df_with_times(100)
    found = [{"start": 30, "window": 20, "score": 0.9, "kind": "symmetric",
              "breakout": "up", "outcome": {"change": -0.005}}]
    text = coin_report("XRPUSDT", df, [("Треугольник", "lines", found)])
    assert "пробой вверх — ложный (↓ -0.5%)" in text
    assert "FVG" not in text                                       # пустые не показываем


def test_split_long_message():
    from bot import split_text
    text = "\n\n".join(f"блок {i} " + "x" * 900 for i in range(10))
    parts = split_text(text, limit=4000)
    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert "\n\n".join(parts) == text
