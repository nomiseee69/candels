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
    items = [{"name": "Треугольник", "score": 0.97, "start": 10, "window": 40, "touches": 6,
              "upper": (10, 110.0, 49, 104.0), "lower": (10, 90.0, 49, 98.0)}]
    code = pine_script("BTCUSDT", "4h", [(df, items, False)])
    t10 = int(df["open_time"].iloc[10].timestamp() * 1000)
    assert code.startswith("//@version=5")
    assert 'indicator("Pattern Finder v49 · BTCUSDT 4H", overlay=true' in code
    assert f"line.new({t10}, 110," in code
    assert code.count("line.new(") == 2
    assert "extend.right" not in code                 # линии не уходят в бесконечность
    # вершина: 104 - 6/39*k = 98 + 8/39*k → k ≈ 16.7 свечи после конца, цена 104 - 2.57 = 101.43
    assert code.count("101.4285714286") == 3         # обе линии и подпись — в одной точке
    assert '"Треугольник"' in code and "касаний" not in code and "color.blue" in code   # только название


def test_past_figures_are_marked_and_explained():
    from pine import pine_help
    df = df_with_times()
    past = [{"name": "Канал", "score": 0.9, "start": 5, "window": 20, "touches": 5,
             "upper": (5, 110.0, 24, 108.0), "lower": (5, 100.0, 24, 98.0), "outcome": {"change": -0.02}}]
    now = [{"name": "FVG", "score": 0.9, "start": 45, "window": 5, "kind": "bullish_fvg",
            "zone": (100.0, 101.0), "filled_at": None}]
    groups = [(df, past, True), (df, now, False)]
    code = pine_script("BTCUSDT", "4h", groups)
    assert '"Канал"' not in code          # прошлая (уже отработанная) фигура пользы не несёт — не рисуем
    assert 'show_history = input.bool(true' in code           # полезная история видна сразу
    text = pine_help("BTCUSDT", "4h", groups)
    assert "Сейчас найдено: FVG" in text


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
    assert code.count("box.new(") == 1                                  # закрытый FVG уже отработан — не рисуем
    assert 'extend=extend.right' in code and 'text="FVG"' in code      # открытый — вправо
    assert f"{t12}, 95, xloc=xloc.bar_time" not in code


def test_pine_history_shape_colored_by_outcome():
    df = df_with_times()
    df.loc[20, "high"] = 130
    df.loc[25, "low"] = 70
    items = [{"name": "Двойное дно", "score": 0.9, "start": 15, "window": 20,
              "outcome": {"change": 0.03}}]
    code = pine_script("ETHUSDT", "1d", [(df, items, True)])
    assert '"Двойное дно"' not in code               # прошлые фигуры не рисуем


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
              "zone": (100.0 + i, 101.0 + i), "filled_at": None} for i in range(550)]
    code = pine_script("BTCUSDT", "1h", [(df, items, True)])
    assert code.count("box.new(") == 2                   # зоны далеко от цены — шум: только ближайшие у цены
    closed = [{**it, "filled_at": it["start"] + 1} for it in items]
    assert pine_script("BTCUSDT", "1h", [(df, closed, True)]).count("box.new(") == 0   # отработанные не рисуем


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
    assert "рано судить" in text and "↑" in text and "%" not in text
    assert "симметричный, пробой вниз (↓)" in text


def test_false_breakout_is_marked():
    df = df_with_times(100)
    found = [{"start": 30, "window": 20, "score": 0.9, "kind": "symmetric",
              "breakout": "up", "outcome": {"change": -0.005}}]
    text = coin_report("XRPUSDT", df, [("Треугольник", "lines", found)])
    assert "пробой вверх — ложный (↓)" in text
    assert "FVG" not in text                                       # пустые не показываем


def test_split_long_message():
    from bot import split_text
    text = "\n\n".join(f"блок {i} " + "x" * 900 for i in range(10))
    parts = split_text(text, limit=4000)
    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert "\n\n".join(parts) == text


def test_history_shows_unworked_zones_and_only_last_figures():
    from pine import _nearest_alive, worth_showing
    channels = [{"name": "Канал", "start": s, "window": 20, "upper": (s, 2, s + 19, 2), "lower": (s, 1, s + 19, 1)}
                for s in (0, 30, 60)]
    bos = [{"name": "BOS", "kind": "bos", "direction": "up", "level": 1.0, "start": s, "window": 5}
           for s in range(0, 50, 10)]
    kept = worth_showing(channels + bos)
    assert [it["start"] for it in kept if it["name"] == "Канал"] == []           # прошлые фигуры — нет
    assert [it["start"] for it in kept if it["name"] == "BOS"] == [40]           # один последний слом
    # зоны: только не задетые и рядом с ценой, по 2 сверху и снизу
    df = df_with_times(60)
    price = float(df["close"].iloc[-1])
    zone = lambda lo, filled=None: {"name": "FVG", "kind": "bullish_fvg", "zone": (lo, lo + 0.5), "start": 3,
                                    "window": 3, "filled_at": filled}
    zones = [zone(price * k) for k in (1.01, 1.02, 1.03, 0.99, 0.98, 0.97, 1.5)] + [zone(price * 1.005, 12)]
    near = _nearest_alive(df, zones)
    assert len(near) == 4 and all(it["filled_at"] is None for it in near)
    assert all(abs(it["zone"][0] / price - 1) < 0.025 for it in near)          # далёкая (×1.5) — нет


def test_channel_and_higher_timeframe_win_over_triangles():
    import scanner
    from pine import pine_script

    def fig(kind, start, score=0.9):
        return {"name": "Канал" if kind.startswith("channel") else "Треугольник", "kind": kind, "score": score,
                "start": start, "window": 20, "touches": 6,
                "upper": (start, 110.0, start + 19, 108.0), "lower": (start, 100.0, start + 19, 98.0)}

    df_1h, df_4h = df_with_times(60), df_with_times(60)
    df_1h["open_time"] = pd.date_range("2026-10-01", periods=60, freq="h", tz="UTC")
    cards = [{"symbol": "X", "interval": "1h", "df": df_1h, "history": False, "caption": "",
              "items": [fig("symmetric", 35), fig("ascending", 5)]},
             {"symbol": "X", "interval": "4h", "df": df_4h, "history": False, "caption": "",
              "items": [fig("channel_down", 30), fig("symmetric", 0)]}]
    coin = scanner.group_by_coin(cards)[0]
    kept = [(f["interval"], it["kind"]) for f in coin["frames"] for it in f["items"]]
    assert kept[0] == ("4h", "channel_down") or ("4h", "channel_down") in kept      # канал старшего ТФ — первым
    assert len(kept) <= 2
    # прошлые фигуры: не больше двух на графике, канал впереди треугольников
    past = [fig("symmetric", s) for s in (0, 20, 40)] + [fig("channel_up", 10)]
    code = pine_script("X", "4h", [(df_4h, past, True)])
    assert code.count('"Канал"') == 0 and code.count('"Треугольник"') == 0       # прошлые фигуры не рисуем
