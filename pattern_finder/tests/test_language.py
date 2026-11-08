"""Описание паттерна словами: бот понимает текст трейдера и ищет именно такой паттерн."""

import language
from market import candles_to_df
from test_eye import fake_market


def test_order_block_description_is_understood():
    spec = language.parse("бычий OB на 4H, обязательно снял ликвидность под лоями, с объёмом, "
                          "сразу поглощение, желательно в дисконте, не тестировался", "Order Block")
    assert set(spec["require"]) == {"liquidity", "volume", "engulf", "fresh"}
    assert spec["prefer"] == ["discount"] and spec["forbid"] == []
    assert spec["timeframes"] == ["4h"] and spec["direction"] == "up" and spec["unknown"] == []
    text = "\n".join(language.describe(spec))
    assert "обязательно" in text and "желательно" in text and "4H" in text and "%" not in text


def test_negations_relations_and_numbers():
    spec = language.parse("медвежий ордер блок, резкий импульс вниз, не против тренда, без FVG", "Order Block")
    assert "impulse" in spec["require"] and "trend" in spec["require"]       # «не против тренда» = по тренду
    assert spec["forbid"] == ["imbalance"] and spec["direction"] == "down"
    spec = language.parse("FVG внутри OB после BOS, без касаний, 1D и 4H", "FVG")
    assert {"inside_ob", "structure", "fresh"} <= set(spec["require"])      # «без касаний» = свежий
    assert set(spec["timeframes"]) == {"1d", "4h"}
    spec = language.parse("равные хаи от 3 касаний, ликвидность ещё не снята", "EQH / EQL")
    assert spec["require"] == ["untapped"] and spec["min_touches"] == 3     # к EQH «ликвидность» — своё понятие
    spec = language.parse("канал с 5 касаниями, лучше крупный, цвет красный", "Канал")
    assert spec["min_touches"] == 5 and spec["prefer"] == ["big"] and spec["unknown"] == ["цвет красный"]


def test_check_filters_findings_by_description():
    df = candles_to_df(fake_market(9, 800))
    from blocks import all_blocks
    blocks = [b for b in all_blocks(df) if b["family"] == "ob"]
    with_fvg = [b for b in blocks if b.get("imbalance")]
    without = [b for b in blocks if not b.get("imbalance")]
    assert with_fvg and without
    need = language.parse("OB обязательно с FVG", "Order Block")
    forbid = language.parse("OB без FVG", "Order Block")
    assert language.check(need, df, dict(with_fvg[0]))[0] and not language.check(need, df, dict(without[0]))[0]
    assert language.check(forbid, df, dict(without[0]))[0] and not language.check(forbid, df, dict(with_fvg[0]))[0]
    only_4h = language.parse("OB только на 4H", "Order Block")
    assert not language.check(only_4h, df, dict(blocks[0]), "1h")[0]
    assert language.timeframes([only_4h]) == {"4h"} and language.timeframes([only_4h, {}]) is None
    bull = language.parse("бычий OB", "Order Block")
    for b in blocks[:6]:
        assert language.check(bull, df, dict(b))[0] == b["kind"].startswith("bullish")


def test_scan_finds_only_described_patterns():
    import scanner
    candles = fake_market(10, 1000)
    plain, _ = scanner.scan_user([{"name": "FVG"}], ["4h", "1h"], lambda s, i, n: candles, ["AUSDT"],
                                 {"quality": "off", "fvg": "all"})
    spec = language.parse("бычий FVG, только 4H", "FVG")
    described, _ = scanner.scan_user([{"name": "FVG", "spec": spec}], ["4h", "1h"], lambda s, i, n: candles,
                                     ["AUSDT"], {"quality": "off", "fvg": "all"})
    for card in described:
        assert card["interval"] == "4h"
        assert all(it["kind"] == "bullish_fvg" for it in card["items"])
    assert len(described) <= len(plain)


def test_bot_description_flow(tmp_path):
    import bot
    from test_menus import Recorder, press, use_storage
    import storage
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        pid = storage.add_pattern(5, "Order Block", None, note="без скриншота", builtin=True)
        press(f"describe:{pid}")
        assert bot.describing[5] == pid and "Опиши словами" in rec.sent[-1][0]
        bot.handle_message({"chat": {"id": 5}, "text": "OB обязательно с объёмом и снял ликвидность, только 4H"})
        text = rec.sent[-1][0]
        assert "Понял твоё описание" in text and "обязательно" in text and "4H" in text
        saved = storage.get_pattern(5, pid)
        assert set(saved["spec"]["require"]) == {"volume", "liquidity"} and saved["spec"]["timeframes"] == ["4h"]
        assert 5 not in bot.describing
    finally:
        rec.restore()
        storage.FILE = original
        bot.describing.clear()


def test_free_wording_and_slang():
    spec = language.parse("должен определяться BOS только у последней свечи которую если пересечь "
                          "то меняет структура рынка", "BOS")
    assert spec["require"] == ["key"] and spec["unknown"] == []
    spec = language.parse("имба которая ещё не заполнена рядом с ценой", "FVG")
    assert "near_price" in spec["require"] and "fresh" in spec["require"] and spec["forbid"] == []
    assert language.parse("не должно быть FVG", "Order Block")["forbid"] == ["imbalance"]


def test_merge_keeps_user_and_learned():
    mine = language.parse("OB с объёмом, только 4H", "Order Block")
    learned = {"require": ["liquidity"], "prefer": [], "forbid": ["imbalance"]}
    both = language.merge(mine, learned)
    assert set(both["require"]) == {"volume", "liquidity"} and both["forbid"] == ["imbalance"]
    assert both["timeframes"] == ["4h"]


def test_trader_slang():
    spec = language.parse("ОБ после сбора ликвы, с большим объемом, следующая свеча перекрывает тело", "Order Block")
    assert set(spec["require"]) == {"liquidity", "volume", "engulf"} and spec["unknown"] == []
    assert language.parse("фвг не перекрытый", "FVG")["require"] == ["fresh"]
    assert "compression" not in language.parse("канал восходящий", "Канал")["require"]
    assert "confirmed" not in language.parse("боковик, ложный пробой вниз", "Range")["require"]
    assert language.parse("мощные зоны на старших таймфреймах", "Order Block")["timeframes"] == ["12h", "1d", "1w"]


def test_every_condition_is_checked_on_every_pattern():
    """Каждое условие, которое бот принимает для паттерна, на нём действительно проверяется
    (а не молча пропускается), и с описанием не приходит ничего, что ему противоречит."""
    import scanner
    from catalog import ALL_PATTERNS
    from templates import template_shape
    settings = {"quality": "off", "fvg": "all", "ob_volume": False}
    known = {name: set() for name in ALL_PATTERNS}
    for seed in range(25):
        df = candles_to_df(fake_market(seed, 1000))
        for name in ALL_PATTERNS:
            example = {"name": name, "builtin": True}
            if name in scanner.SHAPE_PATTERNS:
                example["shape"] = template_shape(name)
            for _, detect in scanner.build_detectors([example], settings)[0]:
                item = detect(df)
                if item:
                    known[name] |= {k for k, v in language.facts(df, {**item, "name": name}).items()
                                    if v is not None and not k.startswith("_")}
    for name, keys in known.items():
        if not keys:
            continue                            # на синтетике паттерн не встретился (ГиП) — проверен экзаменом скринов
        group = language.GROUPS[name]
        needed = {k for k, (_, _, _, groups) in language.CONCEPTS.items() if group in groups}
        assert needed <= keys, (name, needed - keys)


def test_description_controls_figures_flags_and_range():
    import scanner
    cases = [("Канал", "канал, закрытие телом за границей, потом ретест"),
             ("Order Block", "ОБ, частично отработанный"), ("Флаг", "флаг строго по тренду")]
    for name, text in cases:
        spec = language.parse(text, name)
        for seed in range(15):
            candles = fake_market(seed, 1000)
            df = candles_to_df(candles)
            cards, _ = scanner.scan_user([{"name": name, "builtin": True, "spec": spec}], ["4h"],
                                         lambda s, i, n, c=candles: c, ["A"], {"quality": "off", "ob_volume": False})
            for it in (it for c in cards for it in c["items"]):
                assert language.check(spec, df, dict(it))[0], (name, seed)
    assert language.parse("вымпел против тренда не нужен", "Вымпел")["require"] == ["trend"]


def test_timeframe_ranges_and_figure_slope():
    """«4ч и выше», «до 4ч», «падающий клин» — частые формулировки трейдеров."""
    import language
    assert language.parse("имба только 4ч и выше", "FVG")["timeframes"] == ["4h", "12h", "1d", "1w"]
    assert language.parse("ОБ от 12ч", "Order Block")["timeframes"] == ["12h", "1d", "1w"]
    assert language.parse("бос на 4h и ниже", "BOS")["timeframes"] == ["1h", "4h"]
    assert language.parse("fvg 4h", "FVG")["timeframes"] == ["4h"]
    wedge = language.parse("падающий клин", "Клин")
    assert wedge["direction"] == "down" and wedge["slope"]
    assert any("нисходящая" in line for line in language.describe(wedge))
    # наклон фигуры, а не «медвежий»: падающий клин = falling_wedge, восходящий не подходит
    import pandas as pd
    df = pd.DataFrame({"open": [1.0] * 30, "high": [1.1] * 30, "low": [0.9] * 30, "close": [1.0] * 30})
    falling = {"kind": "falling_wedge", "start": 5, "end": 25, "upper": (5, 1.1, 25, 1.0), "lower": (5, 0.9, 25, 0.95)}
    rising = {**falling, "kind": "rising_wedge"}
    assert language.check(wedge, df, falling)[0] is True
    assert language.check(wedge, df, rising)[0] is False
