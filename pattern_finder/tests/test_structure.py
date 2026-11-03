"""Структура рынка: BOS, CHoCH, равные хаи/лои и значимые FVG."""

import numpy as np
import pandas as pd

from smc import find_all_fvg, find_fvg
from structure import equal_levels, find_break, find_equal, structure_events


def path(points, spread=0.2):
    """Свечи вдоль ломаной: points = [(номер свечи, цена), ...]."""
    xs, ys = zip(*points)
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + spread,
        "low": np.minimum(opens, closes) - spread,
        "close": closes,
    })


ZIGZAG = [(0, 100), (10, 110), (20, 104), (30, 115), (40, 108), (50, 120), (65, 100), (70, 100)]


def test_bos_then_choch():
    events = structure_events(path(ZIGZAG))
    kinds = [(e["kind"], e["direction"]) for e in events]
    assert kinds == [("bos", "up"), ("bos", "up"), ("choch", "down")]
    first, second, choch = events
    # у вершины две свечи с одинаковым high (10 и 11) — свингом считается последняя
    assert first["start"] == 11 and abs(first["level"] - 110.2) < 1e-9
    assert second["start"] == 31
    assert choch["start"] == 41 and abs(choch["level"] - 107.8) < 1e-9
    assert "смена тренда" in choch["label"]
    assert all(e["score"] >= 0.85 for e in events)


def test_break_is_known_only_after_close():
    df = path(ZIGZAG)
    choch = structure_events(df)[-1]
    # на свече слома и сразу после — алерт есть; через 10 свечей — уже не свежий
    assert find_break(df.iloc[:choch["end"] + 1], "choch") is not None
    assert find_break(df.iloc[:choch["end"] + 3], "choch") is not None
    assert find_break(df, "choch") is None
    assert find_break(df.iloc[:choch["end"] + 1], "bos") is None


def test_equal_highs_unswept():
    df = path([(0, 100), (10, 110), (20, 104), (30, 110.05), (40, 105)])
    levels = [lv for lv in equal_levels(df) if lv["kind"] == "eqh"]
    assert len(levels) == 1
    eqh = levels[0]
    assert eqh["touches"] == [11, 31] and eqh["swept_at"] is None
    assert abs(eqh["level"] - 110.25) < 1e-9
    assert "ликвидность не снята" in eqh["label"] and "2 касания" in eqh["label"]
    found = find_equal(df)
    assert found is not None and found["kind"] == "eqh"


def test_equal_highs_swept():
    df = path([(0, 100), (10, 110), (20, 104), (30, 110.05), (40, 105), (50, 112), (55, 109)])
    eqh = [lv for lv in equal_levels(df) if lv["kind"] == "eqh"][0]
    assert eqh["swept_at"] is not None and eqh["swept_at"] > 40
    assert "ликвидность снята" in eqh["label"]
    assert find_equal(df) is None or find_equal(df)["kind"] != "eqh"


def test_not_equal_when_level_taken_between():
    # между касаниями цена уже ходила выше — ликвидность сняли, это не равные хаи
    df = path([(0, 100), (10, 110), (15, 104), (20, 113), (25, 104), (35, 110.05), (45, 105)])
    pairs = [lv["touches"] for lv in equal_levels(df) if lv["kind"] == "eqh"]
    assert all(10 not in p and 11 not in p for p in pairs)


def test_different_highs_are_not_equal():
    df = path([(0, 100), (10, 110), (20, 104), (30, 112), (40, 105)])
    assert [lv for lv in equal_levels(df) if lv["kind"] == "eqh"] == []


# --- значимые FVG ------------------------------------------------------------------------------
def flat_rows(n, price=100.0):
    return [(price, price + 0.5, price - 0.5, price)] * n


IMPULSE = [
    (100, 100.5, 99.5, 100.2),
    (100.2, 103.0, 100.1, 102.8),   # импульс: тело 2.6 при ATR ≈ 1
    (102.8, 104.0, 102.0, 103.5),   # дыра 100.5–102.0
]


def frame(rows, volume=None):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    if volume is not None:
        df["volume"] = volume
    return df


def test_big_gap_without_volume_or_structure_is_not_significant():
    df = frame(flat_rows(20) + IMPULSE + flat_rows(3, 103.5))
    gap = [f for f in find_all_fvg(df) if f["kind"] == "bullish_fvg"][0]
    assert not gap["significant"]
    assert find_fvg(df) is not None                       # режим «все» — находим
    assert find_fvg(df, significant_only=True) is None    # режим «значимые» — отсеяли


def test_volume_alone_is_not_enough():
    # большая дыра на объёме, но слома структуры нет (рынок стоял ровно) — не значимый
    rows = flat_rows(20) + IMPULSE + flat_rows(3, 103.5)
    volume = [10.0] * len(rows)
    volume[21] = 40.0
    gap = [f for f in find_all_fvg(frame(rows, volume)) if f["kind"] == "bullish_fvg"][0]
    assert not gap["significant"]


def breaking_rows():
    """Хай 102.5 на свече 10 — свинг; импульс на свече 27 закрывается выше него."""
    rising = path([(0, 100), (10, 102.3), (20, 100), (25, 100)]).to_numpy().tolist()
    return [tuple(r) for r in rising] + [
        (100, 100.4, 99.8, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),
    ]


def test_break_without_volume_is_not_significant():
    rows = breaking_rows() + flat_rows(2, 103.5)
    quiet = [10.0] * len(rows)                           # объём импульса как у всех — неликвидно
    gaps = [f for f in find_all_fvg(frame(rows, quiet)) if f["kind"] == "bullish_fvg"]
    assert gaps and not gaps[0]["significant"]
    loud = list(quiet)
    loud[27] = 30.0
    gaps = [f for f in find_all_fvg(frame(rows, loud)) if f["kind"] == "bullish_fvg"]
    assert gaps and gaps[0]["significant"] and "объём" in gaps[0]["label"]


def test_close_beyond_middle_works_out_significant_fvg():
    # дыра 100.4–102.0, середина 101.2: свеча закрылась на 101.0 — зона отработана
    rows = breaking_rows() + [(103.5, 103.6, 100.9, 101.0)] + flat_rows(2, 101.5)
    df = frame(rows)
    strict = [f for f in find_all_fvg(df, include_filled=True, significant_only=True)
              if f["kind"] == "bullish_fvg"]
    assert strict and strict[0]["filled_at"] == 29 and "отработан" in strict[0]["label"]
    assert find_fvg(df, significant_only=True) is None or find_fvg(df, significant_only=True)["kind"] != "bullish_fvg"
    # в режиме «все» зона ещё открыта: до низа 100.4 цена не дошла
    loose = [f for f in find_all_fvg(df) if f["kind"] == "bullish_fvg"]
    assert loose and loose[0]["filled_at"] is None


def test_impulse_breaking_structure_is_significant():
    # хай 102.5 на свече 10 — свинг; импульс закрывается выше него → слом структуры
    rising = path([(0, 100), (10, 102.3), (20, 100), (25, 100)]).to_numpy().tolist()
    rows = [tuple(r) for r in rising] + [
        (100, 100.4, 99.8, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),
    ] + flat_rows(2, 103.5)
    gaps = [f for f in find_all_fvg(frame(rows)) if f["kind"] == "bullish_fvg"]
    assert gaps and gaps[0]["significant"] and gaps[0]["broke_structure"]
    assert "слом структуры" in gaps[0]["label"]


def test_weak_middle_candle_is_not_significant():
    # дыра есть, но средняя свеча с длинными тенями и маленьким телом — не импульс
    rows = flat_rows(20) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 104.0, 99.9, 100.9),
        (100.9, 103.0, 101.5, 102.5),
    ] + flat_rows(2, 102.5)
    volume = [10.0] * len(rows)
    volume[21] = 40.0
    gaps = [f for f in find_all_fvg(frame(rows, volume)) if f["kind"] == "bullish_fvg"]
    assert gaps and not gaps[0]["significant"]


# --- подключение к сканеру, разметке и настройкам ----------------------------------------------
def with_times(df, freq="4h"):
    df = df.copy()
    df["open_time"] = pd.date_range("2026-10-01", periods=len(df), freq=freq, tz="UTC")
    return df


def test_scanner_finds_choch_and_combos_only():
    import scanner
    df = with_times(path(ZIGZAG))
    choch_end = structure_events(df)[-1]["end"]
    cut = df.iloc[:choch_end + 1].reset_index(drop=True)
    raw = [[int(t.timestamp() * 1000), o, h, lo, c, 1.0] for t, o, h, lo, c in
           zip(cut["open_time"], cut["open"], cut["high"], cut["low"], cut["close"])]
    raw = [raw[0]] * (120 - len(raw)) + raw if len(raw) < 120 else raw     # минимум свечей для поиска
    fetch = lambda s, i, n: raw
    patterns = [{"name": "CHoCH"}, {"name": "BOS"}]
    cards, skipped = scanner.scan_user(patterns, ["4h"], fetch, ["AAAUSDT"])
    assert skipped == [] and len(cards) == 1
    item = cards[0]["items"][0]
    assert item["name"] == "CHoCH" and "CHoCH вниз" in item["label"]
    only = scanner.scan_user(patterns, ["4h"], fetch, ["AAAUSDT"],
                             {"fvg": "significant", "combos_only": True})[0]
    assert only == []                                  # одна находка — не комбинация


def test_pine_draws_structure_and_liquidity():
    from pine import pine_script
    df = with_times(path([(0, 100), (10, 110), (20, 104), (30, 110.05), (40, 105)]))
    eq = [lv for lv in equal_levels(df) if lv["kind"] == "eqh"][0]
    zig = with_times(path(ZIGZAG))
    events = structure_events(zig)
    code = pine_script("X", "4h", [(df, [{**eq, "name": "EQH / EQL"}], True),
                                   (zig, [{**e, "name": "BOS"} for e in events], True)])
    assert code.count("line.new(") == 4
    assert "style=line.style_dotted" in code and "extend=extend.right" in code   # ликвидность не снята
    assert '"EQH $$$"' in code
    assert code.count("style=line.style_dashed") == 3
    assert '"CHoCH"' in code and code.count('"BOS"') == 2


def test_history_texts_for_structure_and_liquidity():
    from history import coin_report, history_equal, history_structure, summarize
    zig = with_times(path(ZIGZAG + [(80, 95)]))
    events = history_structure(zig, "bos")
    assert len(events) == 2 and all(e["outcome"] is not None for e in events)
    assert "в сторону слома в" in summarize("BOS", "structure", events)
    df = with_times(path([(0, 100), (10, 110), (20, 104), (30, 110.05), (40, 105)]))
    levels = history_equal(df)
    text = coin_report("AAAUSDT", df, [("BOS", "structure", events), ("EQH / EQL", "eq", levels)])
    assert "• BOS ×2:" in text and "вверх" in text
    assert "• EQH / EQL: 1 уровень, не снято 1" in text and "ближайший: равные хаи" in text
    assert "ликвидность снята у 0" in summarize("EQH / EQL", "eq", levels)


def test_settings_toggle(tmp_path):
    import storage
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    try:
        assert storage.get_settings(1) == {"fvg": "significant", "combos_only": False, "similar_only": False, "learn": True, "confluence_only": False, "ob_volume": True}
        assert storage.toggle_setting(1, "fvg")["fvg"] == "all"
        assert storage.toggle_setting(1, "combos_only")["combos_only"] is True
        assert storage.toggle_setting(1, "fvg")["fvg"] == "significant"
        assert storage.get_settings(1) == {"fvg": "significant", "combos_only": True, "similar_only": False, "learn": True, "confluence_only": False, "ob_volume": True}
    finally:
        storage.FILE = original
