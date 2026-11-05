import numpy as np
import pandas as pd

from smc import find_all_fvg, find_fvg


def candles(rows):
    """rows: список (open, high, low, close)."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def flat(n, price=100.0):
    """Спокойный рынок: свечи с диапазоном ±0.5 вокруг цены."""
    return [(price, price + 0.5, price - 0.5, price)] * n


def test_bullish_fvg_found():
    rows = flat(20) + [
        (100, 100.5, 99.5, 100.2),     # свеча 1: high 100.5
        (100.2, 103.0, 100.1, 102.8),  # импульс вверх
        (102.8, 104.0, 102.0, 103.5),  # свеча 3: low 102.0 > 100.5 → дыра 100.5–102.0
    ] + [(103.5, 104.0, 103.0, 103.6)] * 5
    found = find_fvg(candles(rows))
    assert found is not None
    assert found["kind"] == "bullish_fvg"
    assert found["zone"] == (100.5, 102.0)
    assert found["filled"] == 0
    assert found["score"] >= 0.85


def test_bearish_fvg_found():
    rows = flat(20) + [
        (100, 100.5, 99.5, 99.8),      # свеча 1: low 99.5
        (99.8, 99.9, 97.0, 97.2),
        (97.2, 98.0, 96.0, 96.5),      # свеча 3: high 98.0 < 99.5 → дыра 98.0–99.5
    ] + [(96.5, 97.0, 96.0, 96.4)] * 5
    found = find_fvg(candles(rows))
    assert found is not None and found["kind"] == "bearish_fvg"
    assert found["zone"] == (98.0, 99.5)


def test_filled_fvg_is_ignored():
    rows = flat(20) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),
        (103.5, 103.6, 100.0, 100.3),  # цена вернулась ниже 100.5 — дыра закрыта
    ] + flat(3, 100.3)
    kinds = [f["kind"] for f in find_all_fvg(candles(rows))]
    # бычий FVG закрыт; резкое падение само образовало новый медвежий — это верно
    assert "bullish_fvg" not in kinds


def test_partial_fill_is_reported():
    rows = flat(20) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),
        (103.5, 103.6, 101.7, 101.9),  # залезла в дыру на 20% — ещё живой
    ]
    found = find_fvg(candles(rows))
    assert found is not None
    assert abs(found["filled"] - 0.2) < 1e-9
    assert found["distance"] == 0          # цена сейчас внутри зоны
    deeper = rows[:-1] + [(103.5, 103.6, 101.2, 101.9)]    # тенью до середины — отработан
    assert find_fvg(candles(deeper)) is None or find_fvg(candles(deeper))["kind"] != "bullish_fvg"


def test_tiny_gap_is_noise():
    rows = flat(20) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 100.8, 100.1, 100.6),
        (100.6, 101.0, 100.55, 100.9),  # дыра 0.05 при ATR ≈ 1 — шум
    ]
    assert find_fvg(candles(rows)) is None


def old_gap(last_price, candles_after=40):
    """FVG 100.5–102.0, образовавшийся давно; потом цена стоит на last_price."""
    p = last_price
    return flat(5) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),
    ] + [(p, p + 0.4, max(p - 0.4, 102.05), p)] * candles_after


def test_old_fvg_far_from_price_is_skipped():
    # цена 103.6 — до зоны 1.5%, а зона старая (40 свечей): не алертим
    assert find_fvg(candles(old_gap(103.6))) is None


def test_old_fvg_near_price_is_reported():
    # цена 102.5 — до зоны 0.5%: старый незакрытый уровень снова важен
    found = find_fvg(candles(old_gap(102.5)))
    assert found is not None
    assert found["label"].startswith("старый бычий FVG")


def test_history_includes_filled_with_time():
    rows = flat(20) + [
        (100, 100.5, 99.5, 100.2),
        (100.2, 103.0, 100.1, 102.8),
        (102.8, 104.0, 102.0, 103.5),      # FVG на свечах 20–22
        (103.5, 103.6, 102.5, 102.8),
        (102.8, 103.0, 100.2, 100.4),      # свеча 24 закрывает дыру
    ]
    gaps = [f for f in find_all_fvg(candles(rows), lookback=100, include_filled=True)
            if f["kind"] == "bullish_fvg"]
    assert len(gaps) == 1
    assert gaps[0]["filled_at"] == 24
    assert find_all_fvg(candles(rows), lookback=100) == [] or all(
        f["kind"] != "bullish_fvg" for f in find_all_fvg(candles(rows), lookback=100))


def test_quiet_market_has_no_fvg():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.05, 100))
    rows = [(c, c + 0.5, c - 0.5, c) for c in close]   # свечи сильно перекрываются
    assert find_all_fvg(candles(rows)) == []
