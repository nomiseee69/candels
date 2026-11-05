"""Обучение на 👍/👎 по-настоящему: оценка сразу меняет поиск."""

import feedback
import language
import scanner
from market import candles_to_df
from test_eye import fake_market


def record(vote, facts, name="Order Block", key=None, interval="4h", direction="up"):
    return {"name": name, "vote": vote, "facts": facts, "key": key or f"X|{interval}|{name}|{id(facts)}",
            "interval": interval, "direction": direction, "features": {}, "factors": None}


def test_concepts_learned_from_votes():
    likes = [record(1, {"liquidity": 1, "volume": v, "imbalance": 0}) for v in (1, 0, 1, 1)]
    dislikes = [record(-1, {"liquidity": 0, "volume": v, "imbalance": 1}) for v in (1, 0, 0)]
    spec = feedback.learned_spec(likes + dislikes)
    assert "liquidity" in spec["require"] and "imbalance" in spec["forbid"]
    text = "\n".join(language.describe(spec))
    assert "обязательно" in text and "нельзя" in text and "%" not in text
    assert feedback.learned_spec(likes[:1]) is None                 # по одной оценке выводов нет


def test_learned_conditions_join_user_description():
    votes = [record(1, {"liquidity": 1}) for _ in range(3)] + [record(-1, {"liquidity": 0}) for _ in range(2)]
    mine = [{"name": "Order Block", "spec": language.parse("OB с объёмом", "Order Block")}, {"name": "FVG"}]
    patterns = feedback.with_learning(mine, votes)
    ob = patterns[0]["spec"]
    assert set(ob["require"]) == {"volume", "liquidity"} and "spec" not in patterns[1]
    assert feedback.with_learning(mine, []) is mine


def test_like_becomes_example_and_dislike_is_never_sent_again():
    candles = fake_market(10, 1000)
    df = candles_to_df(candles)
    fetch = lambda s, i, n: candles
    cards, _ = scanner.scan_user([{"name": "FVG"}], ["4h"], fetch, ["AUSDT"], {"quality": "off", "fvg": "all"})
    item = cards[0]["items"][0]
    key = f"AUSDT|4h|FVG|{int(df['open_time'].iloc[item['start']].timestamp())}"
    votes = [{"name": "FVG", "vote": -1, "key": key, "features": {}, "factors": None}]
    judge = feedback.make_judge(votes)
    again, _ = scanner.scan_user([{"name": "FVG"}], ["4h"], fetch, ["AUSDT"], {"quality": "off", "fvg": "all"},
                                 judge=judge)
    assert not any(it["start"] == item["start"] and it["kind"] == item["kind"]
                   for c in again for it in c["items"])
    from fingerprint import make_fingerprint
    liked = [{"name": "FVG", "vote": 1, "key": "k", "fingerprint": make_fingerprint("FVG", df, item)}]
    patterns = feedback.with_learning([{"name": "FVG"}], liked)
    assert len(patterns) == 2 and patterns[1]["auto"] and scanner._fingerprints(patterns, "FVG")
    assert scanner._specs(patterns, "FVG") == []                  # 👍-пример не отменяет описание


def test_vote_message_tells_what_changed(tmp_path):
    import bot
    import storage
    from test_menus import Recorder, use_storage
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        candles = fake_market(9, 800)
        df = candles_to_df(candles)
        from blocks import all_blocks
        blocks = [b for b in all_blocks(df) if b["family"] == "ob"]
        with_fvg = [b for b in blocks if b.get("imbalance")]
        without = [b for b in blocks if not b.get("imbalance")]
        texts = []
        for i, b in enumerate(with_fvg[:3] + without[:3]):
            coin = {"symbol": f"C{i}USDT", "history": False,
                    "frames": [{"interval": "4h", "df": df, "items": [{**b, "name": "Order Block"}],
                                "history": False}]}
            texts.append(bot.record_vote(5, coin, 0, -1 if b in with_fvg else 1, quiet=True))
        assert texts[0].startswith("👎") and "больше не пришлю" in texts[0]
        assert any("Переучился" in t and "нельзя" in t for t in texts)
        saved = storage.get_feedback(5)
        assert all(r["facts"] for r in saved) and any(r.get("fingerprint") for r in saved if r["vote"] > 0)
        summary = feedback.summary(saved)
        assert "выучил условия" in summary and "%" not in summary
    finally:
        rec.restore()
        storage.FILE = original


def test_wedge_uses_screenshot_description_and_votes_together():
    """Клин: эталон со «скрина» + описание «от 3 касаний сверху и снизу» + 👍/👎 — всё работает вместе."""
    import bot
    from trendlines import FAMILY_BY_PATTERN, find_line_pattern
    ref = None
    for s in range(100, 200):
        df = candles_to_df(fake_market(s, 1000))
        found = find_line_pattern(df, FAMILY_BY_PATTERN["Клин"])
        if found:
            ref = bot.market_example(df, {**found, "name": "Клин"})
            break
    example = {"name": "Клин", "kind": ref["kind"], "reference": ref["reference"]}
    spec = language.parse("от трех касаний сверху и снизу формируется клин", "Клин")
    assert spec["min_touches"] == 3 and spec["touches_each"]
    plain_bad, described = 0, 0
    for s in range(40):
        candles = fake_market(s, 1000)
        fetch = lambda sym, i, n, c=candles: c
        a, _ = scanner.scan_user([example], ["4h"], fetch, ["AUSDT"], {"quality": "off"})
        b, _ = scanner.scan_user([{**example, "spec": spec}], ["4h"], fetch, ["AUSDT"], {"quality": "off"})
        plain_bad += sum(min(language.line_touches(it)) < 3 for c in a for it in c["items"])
        for it in (it for c in b for it in c["items"]):
            described += 1
            assert min(language.line_touches(it)) >= 3 and "similarity" in it     # описание И скрин
    assert plain_bad and described
    liked = [{"name": "Клин", "vote": 1, "key": "k", "example": {"kind": ref["kind"], "reference": ref["reference"]}}]
    assert len([p for p in feedback.with_learning([example], liked) if p.get("reference")]) == 2   # 👍 = пример
