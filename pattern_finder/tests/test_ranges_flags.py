"""Боковик (Range) с профилем объёма, флаг и вымпел, история глубже 1000 свечей,
объём у Order Block."""

import numpy as np
import pandas as pd

import market
from flags import find_flag, history_flags
from ranges import find_range, history_ranges, value_area, volume_profile


def frame(closes, volumes=None, opens=None):
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
    n = len(closes)
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
        "open": opens, "high": np.maximum(opens, closes) + 0.4, "low": np.minimum(opens, closes) - 0.4,
        "close": closes, "volume": np.ones(n) if volumes is None else volumes,
    })


def range_df(poc_at_edge=False):
    """Рост 80 → 100, потом боковик 96–104 на 150 свечей. Объём — у середины (или у верха)."""
    rng = np.random.default_rng(1)
    trend = np.linspace(80, 100, 150) + rng.normal(0, 0.5, 150)
    box = 100 + 4 * np.sin(np.linspace(0, 8 * np.pi, 150)) + rng.normal(0, 0.6, 150)
    closes = np.concatenate([trend, box])
    center = 103.5 if poc_at_edge else 100
    volumes = rng.lognormal(0, 0.3, 300) * (1 + 3 * (np.abs(closes - center) < 1.5))
    return frame(closes, volumes)


def test_range_box_and_poc_in_the_middle():
    df = range_df()
    found = find_range(df)
    assert found is not None and found["kind"] == "range" and found["start"] >= 150
    bottom, top = found["box"]
    assert 94 < bottom < 97 and 103 < top < 106
    assert abs(found["poc"] - 100) < 1.5                         # POC — посередине коробки
    assert found["touches"] >= 4 and "цена внутри" in found["label"]
    assert found["value_area"][0] < found["poc"] < found["value_area"][1]


def test_no_range_when_trending_or_volume_at_the_edge():
    trend = frame(np.linspace(50, 150, 300))
    assert find_range(trend) is None and history_ranges(trend) == []
    assert find_range(range_df(poc_at_edge=True)) is None       # объём у края — не равновесие


def test_range_breakout_and_history():
    df = range_df()
    closes = list(df["close"]) + [106, 108]                      # закрылись над коробкой
    found = find_range(frame(closes, list(df["volume"]) + [3, 3]))
    assert found is not None and found["breakout"] == "up" and "пробой вверх" in found["label"]
    past = history_ranges(df)
    assert past and all(p["start"] >= 140 for p in past)


def test_volume_profile_spreads_volume_by_price():
    profile, centers = volume_profile([10, 10], [0, 8], [10, 4], 0, 10, bins=10)
    assert profile.sum() == 14 and profile[0] == 1.0 and profile[9] == 1.0 + 2.0
    low, high = value_area(np.array([1, 1, 8, 1, 1]), np.arange(5))
    assert low <= 2 <= high


def flag_df(shape="flag", up=True, tail=0):
    """Случайный подход, флагшток 8 свечей по 2.5, тело 16 маленьких свечей, потом выход."""
    rng = np.random.default_rng(2)
    closes = list(100 + np.cumsum(rng.normal(0, 0.3, 80)))
    sign = 1 if up else -1
    closes += [closes[-1] + sign * 2.5 * (i + 1) for i in range(8)]
    top = closes[-1]
    for i in range(16):
        if shape == "flag":
            middle, amp = top - sign * 1.5 - sign * 0.25 * i, 1.4
        else:
            middle, amp = top - sign * 2.0, 2.4 * (1 - i / 18)
        closes.append(middle + amp * (1 if i % 4 < 2 else -1))
    closes += [closes[-1] + sign * 1.5 * (i + 1) for i in range(tail)]
    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    body = slice(88, 104)
    opens[body] = closes[body] - 0.2 * np.sign(closes[body] - np.roll(closes, 1)[body])
    df = frame(closes, opens=opens)
    df["high"] = np.maximum(df["open"], df["close"]) + 0.3
    df["low"] = np.minimum(df["open"], df["close"]) - 0.3
    return df


def test_flag_and_pennant_both_directions():
    for shape in ("flag", "pennant"):
        for up in (True, False):
            found = find_flag(flag_df(shape, up), shape)
            side = "bullish" if up else "bearish"
            assert found is not None and found["kind"] == f"{side}_{shape}", (shape, up)
            assert found["direction"] == ("up" if up else "down") and found["touches"] >= 4
            assert found["pole"][0] < found["upper"][0] and found["retrace"] <= 0.5
            other = "pennant" if shape == "flag" else "flag"
            assert find_flag(flag_df(shape, up), other) is None


def test_flag_breakout_and_history():
    found = find_flag(flag_df("flag", True, tail=3), "flag")
    assert found is not None and "выход вверх" in found["label"]
    past = history_flags(flag_df("flag", True, tail=12), "flag")
    assert len(past) == 1 and past[0]["breakout"] == "up"


def test_pine_draws_range_and_flag():
    from pine import pine_script
    df = range_df()
    rng_item = {**find_range(df), "name": "Range"}
    code = pine_script("X", "4h", [(df, [rng_item], False)])
    assert 'text="Range"' in code and '"POC"' in code
    fdf = flag_df()
    flag = {**find_flag(fdf, "flag"), "name": "Флаг"}
    code = pine_script("X", "4h", [(fdf, [flag], False)])
    assert "color.purple" in code                                    # флагшток


def test_scanner_finds_range_and_flag():
    import scanner
    for name, df in (("Range", range_df()), ("Флаг", flag_df())):
        raw = [[int(t.timestamp() * 1000), o, h, lo, c, v] for t, o, h, lo, c, v in
               zip(df["open_time"], df["open"], df["high"], df["low"], df["close"], df["volume"])]
        raw = [raw[0]] * max(0, 130 - len(raw)) + raw
        cards, skipped = scanner.scan_user([{"name": name}], ["4h"], lambda s, i, n: raw, ["AAAUSDT"])
        assert skipped == [] and cards and cards[0]["items"][0]["name"] == name


def test_get_candles_pages_back_beyond_1000():
    """2500 свечей = 3 запроса: каждый следующий заканчивается перед самой старой свечой."""
    calls = []

    def fake_get(path, params):
        calls.append(params)
        end = params.get("endTime", 10_000_000)
        return [[end - (params["limit"] - 1 - i) * 10, 1, 1, 1, 1, 1] for i in range(params["limit"])]

    saved = market._get
    market._get = fake_get
    try:
        candles = market.get_candles("AAAUSDT", "4h", 2500)
    finally:
        market._get = saved
    times = [c[0] for c in candles]
    assert len(candles) == 2500 and len(calls) == 3 and "endTime" not in calls[0]
    assert times == sorted(times) and len(set(times)) == 2500
    assert calls[1]["endTime"] == times[-999] - 1    # перед самой старой из первого запроса (999 закрытых)
