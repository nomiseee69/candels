"""Обучение на оценках: признаки, правила-пороги, «подходит тебе N%»."""

import pandas as pd

from feedback import (features, fit_probability, learn_rules, make_judge, passes, rule_text,
                      summary)


def frame(n=60, price=100.0):
    times = pd.date_range("2026-10-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open_time": times, "open": price, "high": price + 1,
                         "low": price - 1, "close": price})


def channel(touches, similarity=0.8):
    return {"name": "Канал", "score": 0.9, "start": 10, "window": 50, "touches": touches,
            "violations": 0, "similarity": similarity,
            "upper": (10, 110.0, 59, 104.0), "lower": (12, 100.0, 59, 94.0)}


def record(touches, vote, similarity=0.8):
    return {"name": "Канал", "vote": vote, "features": features(frame(), channel(touches, similarity))}


def test_features_of_figure_and_zone():
    f = features(frame(), channel(6, 0.84))
    assert f["touches"] == 6 and f["similarity"] == 0.84 and f["length"] == 50
    assert abs(f["height_pct"] - 10.0) < 1e-9                  # (110 − 100) / 100
    fvg = {"name": "FVG", "score": 0.9, "start": 50, "window": 10, "kind": "bullish_fvg",
           "zone": (99.0, 100.0), "distance": 0.004, "significant": True}
    g = features(frame(), fvg)
    assert g["size_pct"] == 1.0 and abs(g["distance_pct"] - 0.4) < 1e-9 and g["significant"] == 1


def test_rule_learned_from_votes():
    votes = [record(t, +1) for t in (6, 7, 6, 8, 7)] + [record(t, -1) for t in (4, 5, 4, 5)]
    rules = learn_rules(votes)
    assert rules and rules[0]["feature"] == "touches" and rules[0]["op"] == ">="
    assert 5 < rules[0]["value"] <= 6
    assert rule_text(rules[0]) == "касаний ≥ 6"
    assert passes({"touches": 7}, rules[0]) and not passes({"touches": 5}, rules[0])
    assert passes({}, rules[0])                                 # нет признака — не отсекаем


def test_no_rules_without_enough_or_mixed_votes():
    assert learn_rules([record(6, +1), record(4, -1)]) == []    # мало оценок
    mixed = [record(t, v) for t, v in [(4, 1), (5, -1), (6, 1), (7, -1), (4, -1), (5, 1), (6, -1), (7, 1)]]
    assert learn_rules(mixed) == []                             # 👍 и 👎 перемешаны — границы нет


def test_fit_probability_follows_similar_votes():
    votes = [record(t, +1, 0.9) for t in (7, 8, 7)] + [record(t, -1, 0.3) for t in (4, 4, 5)]
    good = features(frame(), channel(8, 0.88))
    bad = features(frame(), channel(4, 0.35))
    assert fit_probability(votes, good) > 0.8
    assert fit_probability(votes, bad) < 0.2
    assert fit_probability(votes[:3], good) is None             # меньше 5 оценок — не гадаем


def test_judge_filters_and_scores():
    votes = [record(t, +1) for t in (6, 7, 6, 8, 7)] + [record(t, -1) for t in (4, 5, 4, 5)]
    judge = make_judge(votes)
    keep, fit = judge("Канал", frame(), channel(7))
    assert keep and fit > 0.5
    assert judge("Канал", frame(), channel(4)) == (False, None)
    assert make_judge(votes, use_rules=False)("Канал", frame(), channel(4))[0]
    assert judge("FVG", frame(), {"name": "FVG", "score": 0.9, "start": 1, "window": 2,
                                  "zone": (1, 2)}) == (True, None)     # не оценивал — как раньше
    text = summary(votes)
    assert "• Канал: 👍 5, 👎 4" in text and "касаний ≥ 6" in text
