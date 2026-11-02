"""Одна монета — одно сообщение со всеми таймфреймами и один Pine-скрипт на все ТФ."""

import numpy as np
import pandas as pd

import scanner
from pine import pine_help, pine_script


def frame_df(n=60, freq="4h", price=100.0):
    times = pd.date_range("2026-10-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": times, "open": price, "high": price + 1,
                         "low": price - 1, "close": price, "volume": 1.0})


def card(symbol, interval, items, freq):
    df = frame_df(freq=freq)
    return {"symbol": symbol, "interval": interval, "df": df, "items": items, "history": False,
            "caption": scanner.alert_caption(symbol, interval, df, items)}


FVG = {"name": "FVG", "score": 0.95, "start": 50, "window": 10, "kind": "bullish_fvg",
       "zone": (99.0, 100.0), "filled_at": None, "label": "бычий FVG 99–100"}
OB = {"name": "Order Block", "score": 0.9, "start": 40, "window": 15, "kind": "bearish_ob",
      "family": "ob", "zone": (101.0, 102.0), "broken_at": None, "label": "медвежий OB 101–102"}


def test_frames_of_one_coin_are_grouped():
    cards = [card("SOLUSDT", "4h", [FVG], "4h"), card("ETHUSDT", "1h", [FVG], "h"),
             card("SOLUSDT", "1h", [FVG, OB], "h"), card("SOLUSDT", "1d", [OB], "D")]
    coins = scanner.group_by_coin(cards)
    assert [c["symbol"] for c in coins] == ["SOLUSDT", "ETHUSDT"]       # больше находок — выше
    sol = coins[0]
    assert [f["interval"] for f in sol["frames"]] == ["1h", "4h", "1d"]  # от младшего ТФ к старшему
    caption = sol["caption"]
    # FVG ↑ и на 1H, и на 4H — совпадение по таймфреймам
    assert caption.startswith("🔗 SOLUSDT: находки на 3 таймфреймах (1H, 4H, 1D)")
    assert "🔗 Совпадение ТФ:\n• 1H FVG ↑ внутри 4H FVG ↑" in caption
    assert "• FVG ↑ сразу на 1H, 4H" in caption
    assert caption.index("⏱ 1H  🔥 комбинация") < caption.index("⏱ 4H") < caption.index("⏱ 1D")
    assert "interval=60" in caption                                      # ссылка на младший ТФ
    assert coins[1]["caption"] == coins[1]["frames"][0]["caption"]       # один ТФ — как раньше


def test_coin_chart_has_panel_per_timeframe(tmp_path):
    cards = [card("SOLUSDT", "1h", [FVG], "h"), card("SOLUSDT", "4h", [OB], "4h")]
    path = scanner.draw_coin(scanner.group_by_coin(cards)[0], str(tmp_path / "coin.png"))
    from PIL import Image
    width, height = Image.open(path).size
    assert height > width * 0.5                                         # две панели друг под другом


def test_one_pine_for_all_timeframes():
    df_1h, df_4h = frame_df(freq="h"), frame_df(freq="4h")
    groups = [(df_1h, [FVG], False, "1h"), (df_4h, [OB], True, "4h"), (df_4h, [], False, "4h")]
    code = pine_script("SOLUSDT", "1h", groups)
    assert 'indicator("Pattern Finder v18 · SOLUSDT 1H 4H"' in code
    assert 'show_1h = input.bool(true, "1H", group="Таймфреймы")' in code and 'show_4h = input.bool(true, "4H"' in code
    assert "chart_seconds = timeframe.in_seconds()" in code
    assert "    if show_4h and (not only_higher or chart_seconds <= 14400)" in code
    assert 'text="1H FVG"' in code and 'text="4H OB"' in code             # подписи с таймфреймом
    body = code.split("if barstate.islast")[1]
    assert body.index("show_1h") < body.index("show_4h")
    text = pine_help("SOLUSDT", "1h", groups)
    assert "таймфреймы 1H, 4H в одном файле" in text and "Сейчас найдено: 1H: FVG" in text


def test_single_timeframe_script_has_no_switches():
    code = pine_script("SOLUSDT", "4h", [(frame_df(), [FVG], False)])
    assert "only_higher" not in code and "show_4h" not in code and 'text="FVG"' in code


def test_pine_keeps_current_objects_when_over_limit():
    df = frame_df(n=700, freq="h")
    past = [{"name": "FVG", "score": 0.9, "start": i, "window": 2, "kind": "bullish_fvg",
             "zone": (100.0, 101.0), "filled_at": i + 1} for i in range(600)]
    now = [{**FVG, "start": 690}]
    code = pine_script("X", "1h", [(df, past, True, "1h"), (frame_df(freq="4h"), now, False, "4h")])
    assert code.count("box.new(") <= 450
    assert 'text="4H FVG"' in code                                       # текущая находка не выброшена


def test_full_markup_covers_all_user_timeframes(monkeypatch=None):
    calls = []

    def fake_candles(symbol, interval, limit):
        calls.append(interval)
        return [[int(t.timestamp() * 1000), 1, 2, 0.5, 1.5, 1] for t in
                pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")]

    original = scanner.get_candles
    scanner.get_candles = fake_candles
    try:
        coin = scanner.group_by_coin([card("SOLUSDT", "4h", [FVG], "4h")])[0]
        groups = scanner.full_markup([{"name": "FVG"}], coin, ["1h", "4h", "1d"])
    finally:
        scanner.get_candles = original
    assert calls == ["1h", "4h", "1d"]                                   # и ТФ без текущих находок
    assert [(g[2], g[3]) for g in groups] == [(True, "1h"), (True, "4h"), (False, "4h"), (True, "1d")]


def test_declutter_keeps_figure_with_more_touches():
    tri = {"name": "Треугольник", "score": 0.93, "start": 10, "window": 50, "touches": 6,
           "upper": (10, 110, 59, 104), "lower": (12, 90, 59, 98)}
    chan = {**tri, "name": "Канал", "score": 0.85, "touches": 5, "start": 15, "window": 45}
    far = {**tri, "name": "Канал", "start": 200}
    kept = scanner.declutter_figures([FVG, tri, chan, far])
    assert [it["name"] for it in kept] == ["FVG", "Треугольник", "Канал"]
    assert all(it is not chan for it in kept)


# --- чистая разметка ------------------------------------------------------------------------------
def fvg_at(bottom, start=10):
    return {"name": "FVG", "score": 0.9, "start": start, "window": 5, "kind": "bullish_fvg",
            "zone": (bottom, bottom + 0.5), "filled_at": None}


def test_only_nearest_live_zones_are_shown():
    from pine import plan
    df = frame_df()                                     # цена 100
    above = [fvg_at(101 + i, start=10 + i) for i in range(5)]
    below = [fvg_at(95 - i, start=20 + i) for i in range(4)]
    far = [fvg_at(130)]                                 # дальше 15% — не показываем
    layers = plan([(df, above + below + far, True, "4h")])["4h"]
    alive = sorted(it["zone"][0] for _, it in layers["alive"])
    assert alive == [95, 101]                           # по одной ближайшей сверху и снизу
    assert len(layers["past"]) == 8                     # остальные — в скрытой истории


def test_same_zone_on_two_timeframes_is_drawn_once():
    from pine import plan
    zone_1h = fvg_at(101.0)
    zone_4h = {**fvg_at(100.9), "zone": (100.9, 101.6)}
    layers = plan([(frame_df(freq="h"), [zone_1h], False, "1h"), (frame_df(), [zone_4h], False, "4h")])
    assert layers["1h"]["now"] == []                    # остаётся на старшем ТФ
    assert len(layers["4h"]["now"]) == 1


def test_summary_table_in_corner():
    df_1h, df_4h = frame_df(freq="h"), frame_df(freq="4h")
    code = pine_script("SOLUSDT", "1h", [(df_1h, [FVG], False, "1h"), (df_4h, [OB], False, "4h")])
    assert "table.new(position.top_right, 3, 3" in code
    assert 'table.cell(pf_table, 0, 1, "1H"' in code and 'table.cell(pf_table, 2, 2, "Order Block ↓"' in code
    assert code.index("if show_table") > code.index("if show_1h")
