"""ИИ-значимость: метки по исходам, обучение, экзамен и как бот её использует."""

import numpy as np
import pandas as pd

import quality
from market import candles_to_df
from test_eye import fake_market


def frame(closes, spread=0.3):
    closes = np.asarray(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + spread,
                         "low": np.minimum(opens, closes) - spread, "close": closes,
                         "volume": np.ones(len(closes))})


def test_reaction_label_up_before_stop():
    a = quality.Arrays(frame([10] * 30 + [10, 11, 12, 13, 14, 15] + [15] * 30))
    assert quality.reaction(a, 30, True, entry=10, stop=9, target=3) == 1
    a = quality.Arrays(frame([10] * 30 + [9, 8, 7, 12, 14, 16] + [16] * 30))
    assert quality.reaction(a, 30, True, entry=10, stop=9.5, target=3) == 0     # сначала закрылись за зоной
    a = quality.Arrays(frame([10] * 40))
    assert quality.reaction(a, 35, True, entry=10, stop=9, target=3, horizon=20) is None   # исход неизвестен


def test_zone_touch_and_label_without_peeking():
    df = frame([10] * 30 + [12, 14, 16, 15, 13, 11.5, 13, 16, 18] + [18] * 30)
    a = quality.Arrays(df)
    item = {"kind": "bullish_fvg", "start": 29, "end": 31, "zone": (11.0, 12.0)}
    t = quality.first_touch(a, item["zone"], 31, True)
    assert t == 35                                   # low 11.2 ≤ 12 — первое касание зоны
    t2, y = quality.label("fvg", a, item)
    assert t2 == t - 1 and y == 1                    # признаки — по свече ДО касания
    # признаки на момент t2 не зависят от того, что было после
    x = quality.features("fvg", a, item, t2)
    cut = quality.Arrays(df.iloc[:t2 + 1].reset_index(drop=True))
    assert np.allclose(x, quality.features("fvg", cut, item))


def test_family_and_direction():
    assert quality.family_of({"kind": "bearish_fvg"}) == "fvg"
    assert quality.family_of({"kind": "bullish_breaker"}) == "block"
    assert quality.family_of({"kind": "choch"}) == "structure"
    assert quality.family_of({"kind": "eqh"}) == "eq"
    assert quality.family_of({"points": [(0, 1)], "neckline": (0, 1, 2, 1)}) == "shape"
    assert quality.family_of({"upper": 1}) is None
    assert not quality.is_bullish({"kind": "eqh"}) and quality.is_bullish({"kind": "eql"})
    hs = {"points": [(0, 5), (1, 3), (2, 7), (3, 3), (4, 5), (5, 2)], "neckline": (1, 3, 5, 3)}
    assert not quality.is_bullish(hs)                            # ГиП вершиной — медвежья
    assert quality.is_bullish({**hs, "name": "Перевёрнутая ГиП"})


def rows(n, seed, signal=True):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n):
        x = rng.normal(0, 1, 12)
        y = int(x[0] + 0.7 * rng.normal() > 0) if signal else int(rng.random() < 0.4)
        result.append(("fvg", x, None, y, i / n))
    return result


def test_model_turns_on_only_with_real_lift():
    good = quality.fit_family(rows(600, 1), rows(200, 2), use_mlp=False)
    assert good["on"] and good["exam"]["auc"] > 0.7
    assert good["exam"]["strong"] > good["exam"]["base"] > good["exam"]["weak"]
    noise = quality.fit_family(rows(600, 3, False), rows(200, 4, False), use_mlp=False)
    assert not noise["on"]
    assert quality.fit_family(rows(50, 5), rows(200, 6)) is None              # мало случаев


def test_auc():
    assert quality.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert quality.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert quality.auc([1, 1], [0.1, 0.2]) == 0.5


def test_train_all_on_market_and_save_load(tmp_path):
    frames = [candles_to_df(fake_market(i, 1000)) for i in range(4)]
    models = quality.train_all(frames[:3], frames[3:], families=["fvg", "block"], use_mlp=False,
                               say=lambda text: None)
    assert "fvg" in models
    assert not models["fvg"]["on"]                     # случайное блуждание — значимость не угадать
    path = str(tmp_path / "q.json")
    quality.save(models, path)
    try:
        assert quality.load(path) == []
        assert set(quality.MODELS) == set(models)
    finally:
        quality.use({})


def constant_model(family, p, weak=0.3, strong=0.6):
    dims = {"fvg": 12, "block": 15, "structure": 11, "eq": 9, "shape": 10}[family]
    logit = float(np.log(p / (1 - p)))
    return {"type": "logistic", "mean": [0] * dims, "std": [1] * dims, "w": [0] * dims, "b": logit,
            "weak": weak, "strong": strong, "on": True, "exam": {"base": 0.4}}


def test_feature_sizes_match():
    df = candles_to_df(fake_market(7, 600))
    a = quality.Arrays(df)
    seen = set()
    for family, item in quality.candidates(df, ["fvg", "block", "structure", "eq"]):
        if family in seen:
            continue
        seen.add(family)
        size = len(quality.features(family, a, item))
        assert size == len(constant_model(family, 0.5)["w"]), family


def test_quality_shifts_expert_strength():
    import strength
    df = candles_to_df(fake_market(11, 1000))
    fvg = next(it for fam, it in quality.candidates(df, ["fvg"]))
    try:
        quality.use({})
        plain = strength.evaluate(df, dict(fvg))
        quality.use({"fvg": constant_model("fvg", 0.7)})      # история: такие FVG дают реакцию чаще
        assert strength.evaluate(df, dict(fvg)) > plain
        quality.use({"fvg": constant_model("fvg", 0.2)})      # реже обычного
        assert strength.evaluate(df, dict(fvg)) < plain
    finally:
        quality.use({})
