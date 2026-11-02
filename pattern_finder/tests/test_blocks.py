"""Order Block, Breaker Block, Mitigation Block на нарисованных вручную графиках."""

from blocks import all_blocks, find_block
from test_structure import path

# хай 110 → лой 104 → хай 112 → падение ниже 104 (слом вниз) → рост выше 112 (провал)
BREAKER = [(0, 100), (10, 110), (20, 104), (30, 112), (40, 100), (55, 120), (70, 121)]
# то же, но падение остановилось на 106 — выше прошлого лоя 104
MITIGATION = [(0, 100), (10, 110), (20, 104), (30, 112), (40, 106), (55, 120)]
# лой 90 → хай 96 → лой 88 → рост выше 96 (слом вверх) → откат обратно в зону
BULLISH_OB = [(0, 100), (10, 90), (20, 96), (30, 88), (40, 100), (50, 89)]


def by_kind(blocks, kind):
    return [blk for blk in blocks if blk["kind"] == kind]


def test_bearish_ob_then_bullish_breaker():
    blocks = all_blocks(path(BREAKER))
    ob = by_kind(blocks, "bearish_ob")[0]
    # последняя растущая свеча перед падением от хая 112 — свеча 30
    assert ob["start"] == 30 and ob["zone"] == (111.0, 112.2)
    assert ob["broken_at"] is not None                     # цена ушла выше 112 — OB провалился
    breaker = by_kind(blocks, "bullish_breaker")[0]
    assert breaker["start"] == 30 and breaker["end"] == ob["broken_at"]
    assert breaker["broken_at"] is None and "бычий Breaker 111–112.2" in breaker["label"]
    assert by_kind(blocks, "bullish_mitigation") == []


def test_higher_low_gives_mitigation_not_breaker():
    blocks = all_blocks(path(MITIGATION))
    assert by_kind(blocks, "bearish_ob") == []             # слома вниз не было
    mitigation = by_kind(blocks, "bullish_mitigation")
    assert len(mitigation) == 1 and mitigation[0]["zone"] == (111.0, 112.2)
    assert by_kind(blocks, "bullish_breaker") == []


def test_bullish_ob_retest_is_alerted():
    df = path(BULLISH_OB)
    ob = by_kind(all_blocks(df), "bullish_ob")[-1]
    assert ob["start"] == 30 and ob["zone"] == (87.8, 89.0)   # последняя падающая свеча у лоя
    assert ob["touched_at"] is not None and ob["broken_at"] is None
    found = find_block(df, "ob")
    assert found is not None and found["kind"] == "bullish_ob"
    assert "цена в зоне" in found["label"] and "тестировался" in found["label"]


def test_block_is_known_only_after_the_break():
    df = path(BULLISH_OB)
    ob = by_kind(all_blocks(df), "bullish_ob")[-1]
    before = all_blocks(df.iloc[:ob["end"]])               # до свечи слома блока ещё нет
    assert not [blk for blk in by_kind(before, "bullish_ob") if blk["start"] == 30]
    after = all_blocks(df.iloc[:ob["end"] + 1])
    fresh = [blk for blk in by_kind(after, "bullish_ob") if blk["start"] == 30]
    assert fresh and fresh[0]["end"] == ob["end"]
    found = find_block(df.iloc[:ob["end"] + 1], "ob")      # только что сложился → алерт
    assert found is not None and found["start"] == 30


def test_broken_block_is_not_alerted():
    df = path(BREAKER)
    assert all(blk["start"] != 30 for blk in [find_block(df, "ob")] if blk)


def test_ob_with_imbalance_is_marked():
    import pandas as pd
    # широкие свечи (тени ±1) перекрываются — дыр нет
    calm = path(BULLISH_OB, spread=1.0)
    plain = [blk for blk in all_blocks(calm) if blk["kind"] == "bullish_ob"]
    assert plain and not any(blk["imbalance"] for blk in plain)
    # делаем рост после лоя резким: свеча 34 открывается с разрывом над свечой 32 → FVG вверх
    rows = calm.to_dict("records")
    base = rows[32]["close"]
    rows[33] = {"open": base, "high": base + 4.2, "low": base - 0.1, "close": base + 4.0}
    rows[34] = {"open": base + 4.0, "high": base + 5.0, "low": rows[32]["high"] + 1.5, "close": base + 4.8}
    sharp = pd.DataFrame(rows)
    marked = [blk for blk in all_blocks(sharp) if blk["kind"] == "bullish_ob" and blk["imbalance"]]
    assert marked and "(с FVG)" in marked[0]["label"]


# --- подключение к сканеру, истории и разметке ------------------------------------------------
def with_times(df):
    import pandas as pd
    df = df.copy()
    df["open_time"] = pd.date_range("2026-10-01", periods=len(df), freq="4h", tz="UTC")
    return df


def test_scanner_alerts_order_block_retest():
    import scanner
    df = with_times(path(BULLISH_OB))
    raw = [[int(t.timestamp() * 1000), o, h, lo, c, 1.0] for t, o, h, lo, c in
           zip(df["open_time"], df["open"], df["high"], df["low"], df["close"])]
    raw = [raw[0]] * (130 - len(raw)) + raw                # минимум свечей для поиска
    cards, skipped = scanner.scan_user([{"name": "Order Block"}, {"name": "Breaker Block"}],
                                       ["4h"], lambda s, i, n: raw, ["AAAUSDT"])
    assert skipped == [] and len(cards) == 1
    item = cards[0]["items"][0]
    assert item["name"] == "Order Block" and "бычий OB" in item["label"]


def test_pine_draws_blocks():
    from pine import pine_script
    df = with_times(path(BREAKER))
    blocks = [{**blk, "name": "Order Block"} for blk in all_blocks(df)]
    code = pine_script("X", "4h", [(df, blocks, True)])
    assert code.count("box.new(") == len(blocks)
    assert 'text="Breaker"' in code                       # живой breaker у цены — с подписью
    assert code.count('text=""') == len(blocks) - 1       # остальные — история, без подписей
    assert "border_style=line.style_dashed" in code       # breaker — пунктирная рамка
    assert "extend=extend.right" in code                  # действующий блок тянется вправо


def test_history_texts_for_blocks():
    from history import coin_report, history_blocks, summarize
    df = with_times(path(BREAKER))
    obs = history_blocks(df, "ob")
    assert obs and all(blk["family"] == "ob" for blk in obs)
    text = summarize("Order Block", "block", obs)
    assert "пробито" in text and "действует" in text
    report = coin_report("AAAUSDT", df, [("Order Block", "block", obs)])
    assert report.startswith("🪙 AAAUSDT") and "• Order Block:" in report
