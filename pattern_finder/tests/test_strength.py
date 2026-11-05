"""Эксперт силы находок и личная подстройка весов по 👍/👎."""

import numpy as np

import feedback
import scanner
import strength
from blocks import all_blocks
from market import candles_to_df
from test_examples import ob_df
from test_eye import fake_market


def bullish_ob(df):
    return [b for b in all_blocks(df) if b["kind"] == "bullish_ob"][-1]


def test_strong_impulse_with_volume_beats_weak_block():
    df = ob_df(impulse=8, volume=3, seed=2)
    strong = max((b for b in all_blocks(df) if b["kind"] == "bullish_ob"), key=lambda b: b["displacement"])
    # тот же блок, но без резкого импульса, без FVG и без объёма
    weak = {**strong, "displacement": 0.3, "imbalance": False, "volume_ratio": 0.9}
    s1, s2 = strength.evaluate(df, strong), strength.evaluate(df, weak)
    assert s1 > s2
    assert strong["factors"]["impulse"] > weak["factors"]["impulse"]
    assert strong["factors"]["volume"] > weak["factors"]["volume"]
    assert strong["grade"] in ("strong", "normal") and strong["group"] == "zone"
    reasons = strength.strongest(strong)
    assert reasons and set(reasons) <= set(strength.FACTOR_NAMES.values())


def test_every_family_gets_factors():
    import quality
    from catalog import FLAG_SHAPE
    from flags import history_flags
    from history import history_lines
    from ranges import history_ranges
    from trendlines import FAMILY_BY_PATTERN, WINDOWS
    df = candles_to_df(fake_market(21, 1000))
    items = [it for _, it in quality.candidates(df, ["fvg", "block", "structure", "eq"])]
    items += history_flags(df, FLAG_SHAPE["Флаг"]) + history_ranges(df)
    items += history_lines(df, FAMILY_BY_PATTERN["Канал"], WINDOWS, 0.6)
    groups = set()
    for it in items:
        s = strength.evaluate(df, it)
        assert s is not None and 0 <= s <= 1 and it["grade"] in ("strong", "normal", "weak")
        assert all(0 <= v <= 1 for v in it["factors"].values())
        groups.add(it["group"])
    assert {"zone", "structure", "eq", "figure"} <= groups


def test_passes_by_strictness():
    weak, normal, strong = ({"grade": lv} for lv in ("weak", "normal", "strong"))
    assert not strength.passes(weak, {}) and strength.passes(normal, {})
    assert not strength.passes(normal, {"quality": "strict"}) and strength.passes(strong, {"quality": "strict"})
    assert strength.passes(weak, {"quality": "off"})
    assert strength.passes({}, {})                                   # без оценки — не прячем


def votes(seed, n, liked_factor="liquidity"):
    """Трейдер, которому важно только снятие ликвидности: 👍 если оно есть, иначе 👎."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        facts = {k: float(rng.random()) for k in strength.PRIOR["zone"]}
        facts[liked_factor] = float(rng.random() < 0.5)
        records.append({"name": "Order Block", "vote": 1 if facts[liked_factor] else -1, "features": {},
                        "factors": facts, "group": "zone"})
    return records


def test_personal_weights_follow_votes():
    assert feedback.personal_weights(votes(1, 2)) is None            # мало оценок
    personal = feedback.personal_weights(votes(1, 40))
    prior = strength.PRIOR["zone"]
    assert personal.weights["liquidity"] > prior["liquidity"] * 1.5
    more, less, _ = personal.changes()
    assert "снятие ликвидности" in more
    with_sweep = {k: 0.5 for k in prior} | {"liquidity": 1.0}
    without = {k: 0.5 for k in prior} | {"liquidity": 0.0}
    assert personal.logit(with_sweep) - personal.logit(without) > \
        strength.expert_logit("zone", with_sweep) - strength.expert_logit("zone", without)
    # мало оценок — веса почти учебные (выводов по паре кликов не делаем)
    few = feedback.personal_weights(votes(1, 4))
    assert abs(few.weights["liquidity"] - prior["liquidity"]) < abs(personal.weights["liquidity"] - prior["liquidity"])


def test_strictness_adapts_to_votes():
    records = votes(3, 30)
    for r in records:
        r["vote"] = -1                                               # всё не нравится
    personal = feedback.personal_weights(records)
    assert "строже" in personal.changes()[2]
    text = feedback.summary(records)
    assert "отбираю для тебя строже" in text and "%" not in text


def test_judge_uses_personal_weights():
    df = ob_df(impulse=8, volume=3, seed=2)
    item = {**bullish_ob(df), "name": "Order Block"}
    expert = strength.evaluate(df, dict(item))
    records = votes(4, 40)
    for r in records:
        r["vote"] = -1                                               # трейдеру не нравятся OB вообще
    judge = feedback.make_judge(records)
    mine = dict(item)
    judge("Order Block", df, mine)
    assert mine["strength"] < expert


def test_scan_output_has_no_numbers():
    df_candles = fake_market(5, 1000)
    patterns = [{"name": "FVG"}, {"name": "Order Block"}, {"name": "BOS"}, {"name": "EQH / EQL"}]
    cards, _ = scanner.scan_user(patterns, ["4h"], lambda s, i, n: df_candles, ["AUSDT"],
                                 {"quality": "off", "ob_volume": False})
    for card in cards:
        head = card["caption"].split("http")[0]
        assert "%" not in head
        for it in card["items"]:
            assert scanner.describe_item(it) in (f"{it['name']} ↑", f"{it['name']} ↓", it["name"])


def test_weak_findings_are_not_sent():
    df_candles = fake_market(6, 1000)
    patterns = [{"name": "FVG"}, {"name": "Order Block"}, {"name": "BOS"}, {"name": "CHoCH"}]
    every, _ = scanner.scan_user(patterns, ["4h"], lambda s, i, n: df_candles, ["AUSDT"],
                                 {"quality": "off", "ob_volume": False})
    strict, _ = scanner.scan_user(patterns, ["4h"], lambda s, i, n: df_candles, ["AUSDT"],
                                  {"quality": "strict", "ob_volume": False})
    count = lambda cards: sum(len(c["items"]) for c in cards)
    assert count(strict) <= count(every)
    assert all(it["grade"] == "strong" for c in strict for it in c["items"])


def test_markup_keeps_only_not_weak_alive_zones():
    import pine
    df = candles_to_df(fake_market(12, 1000))
    jobs, _ = scanner.build_history_jobs([{"name": "FVG"}], {})
    alive = [it for it in scanner._past_items(jobs, df, [], {}) if pine.is_alive(it)]
    assert alive and all(it["grade"] != "weak" for it in alive)
    everything = [it for it in scanner._past_items(jobs, df, [], {"quality": "off"}) if pine.is_alive(it)]
    assert len(everything) >= len(alive)


def test_rating_keeps_price_level_of_bos_and_eq():
    """Оценка силы не должна портить цену уровня у BOS / CHoCH / EQH / EQL (ключ "level")."""
    import quality
    df = candles_to_df(fake_market(8, 600))
    for family, item in quality.candidates(df, ["structure", "eq"]):
        level = item["level"]
        strength.evaluate(df, item)
        assert item["level"] == level and isinstance(item["level"], float)
        assert item["grade"] in ("strong", "normal", "weak")


def test_market_examples_are_textbook():
    """«👀 Пример с рынка» для OB: только резкий импульс СРАЗУ после блока, объём и FVG."""
    import bot
    good = {"kind": "bullish_ob", "group": "zone", "grade": "strong",
            "factors": {"impulse": 0.8, "volume": 0.6, "imbalance": 1.0}}
    slow = {**good, "factors": {"impulse": 0.3, "volume": 0.6, "imbalance": 1.0}}      # медленный рост
    quiet = {**good, "factors": {"impulse": 0.8, "volume": 0.0, "imbalance": 1.0}}     # объём не на блоке
    assert bot.textbook(good) and not bot.textbook(slow) and not bot.textbook(quiet)
    assert not bot.textbook({**good, "grade": "weak"})
