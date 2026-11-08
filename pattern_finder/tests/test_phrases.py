"""Набор живых трейдерских фраз по ВСЕМ 17 паттернам — как бот понимает их словарём (без нейросети).
Если правка словаря что-то сломает, этот тест покажет, какая фраза перестала пониматься."""

import language

PHRASES = [
    ('FVG', 'свежий фвг на 4ч, рядом с ценой', ['fresh', 'near_price'], [], [], ['4h'], None, None),
    ('FVG', 'медвежья имба которую ещё не заполнили, по тренду', ['fresh', 'trend'], [], [], [], 'down', None),
    ('FVG', 'гэп после сильного импульса, желательно в дисконте', ['impulse'], ['discount'], [], [], None, None),
    ('FVG', 'фвг внутри ордерблока, дневка и выше', ['inside_ob'], [], [], ['1d', '1w'], None, None),
    ('Order Block', 'бычий ОБ после свипа лоёв с объёмом', ['liquidity', 'volume'], [], [], [], 'up', None),
    ('Order Block', 'ордер блок без имбаланса, лучше в премиуме, шорт', [], ['discount'], ['imbalance'], [], 'down', None),
    ('Order Block', 'OB который ещё не тестировали, 1h', ['fresh'], [], [], ['1h'], None, None),
    ('Order Block', 'ОБ после сбора ликвидности над хаями', ['liquidity'], [], [], [], 'down', None),
    ('Breaker Block', 'брейкер после ложного пробоя, медвежий', ['liquidity'], [], [], [], 'down', None),
    ('Breaker Block', 'бычий брейкер с ретестом', ['retest'], [], [], [], 'up', None),
    ('Mitigation Block', 'митигейшн в дисконте на 4ч', ['discount'], [], [], ['4h'], None, None),
    ('Mitigation Block', 'медвежий митигейшн с объёмом', ['volume'], [], [], [], 'down', None),
    ('BOS', 'слом структуры вверх, закрытие телом', ['body_close'], [], [], [], 'up', None),
    ('BOS', 'только последний бос на старших тф', ['key'], [], [], ['12h', '1d', '1w'], None, None),
    ('CHoCH', 'смена характера после снятия ликвидности, медвежий', ['key', 'liquidity'], [], [], [], 'down', None),
    ('CHoCH', 'чох на 1ч, закрепление телом', ['body_close'], [], [], ['1h'], None, None),
    ('EQH / EQL', 'равные хаи, ликвидность ещё не сняли', ['untapped'], [], [], [], None, None),
    ('EQH / EQL', 'равные лои минимум 3 касания', [], [], [], [], None, 3),
    ('Range', 'боковик на 4ч минимум 3 касания сверху и снизу', [], [], [], ['4h'], None, 3),
    ('Range', 'узкий флэт с ложным выносом', ['liquidity', 'small'], [], [], [], None, None),
    ('Двойное дно', 'двойное дно с пробоем шеи на дневке', ['confirmed'], [], [], ['1d'], None, None),
    ('Двойное дно', 'двойное дно с объёмом на втором дне', ['volume'], [], [], [], None, None),
    ('Двойная вершина', 'двойная вершина на 4h, пробита шея', ['confirmed'], [], [], ['4h'], None, None),
    ('Двойная вершина', 'крупная двойная вершина', ['big'], [], [], [], None, None),
    ('Голова и плечи', 'голова и плечи симметричная, с пробоем шеи', ['confirmed', 'symmetry'], [], [], [], None, None),
    ('Голова и плечи', 'ГиП на недельке', [], [], [], ['1w'], None, None),
    ('Перевёрнутая ГиП', 'перевёрнутая гип с объёмом на пробое', ['confirmed', 'volume'], [], [], [], None, None),
    ('Перевёрнутая ГиП', 'обратная голова и плечи на 4ч', [], [], [], ['4h'], None, None),
    ('Треугольник', 'восходящий треугольник с пробоем вверх', ['confirmed'], [], [], [], 'up', None),
    ('Треугольник', 'симметричный треугольник, сужение, 1ч', ['compression'], [], [], ['1h'], None, None),
    ('Клин', 'падающий клин, минимум 3 касания на каждой линии', [], [], [], [], 'down', 3),
    ('Клин', 'восходящий клин на дневке', [], [], [], ['1d'], 'up', None),
    ('Канал', 'нисходящий канал 4 касания', [], [], [], [], 'down', 4),
    ('Канал', 'ровный канал вверх на 12ч', [], [], [], ['12h'], 'up', None),
    ('Флаг', 'бычий флаг после импульса с объёмом', ['impulse', 'volume'], [], [], [], 'up', None),
    ('Флаг', 'медвежий флаг с пробоем', ['confirmed'], [], [], [], 'down', None),
    ('Вымпел', 'вымпел после резкого движения вниз', ['impulse'], [], [], [], 'down', None),
    ('Вымпел', 'бычий вымпел, сужение, 15 минут нет — 1ч', ['compression'], [], [], ['1h'], 'up', None),
]


def test_trader_phrases():
    wrong = []
    for name, text, require, prefer, forbid, timeframes, direction, touches in PHRASES:
        spec = language.parse(text, name)
        got = (sorted(spec["require"]), sorted(spec["prefer"]), sorted(spec["forbid"]), spec["timeframes"],
               spec["direction"], spec["min_touches"])
        if got != (require, prefer, forbid, timeframes, direction, touches) or spec["unknown"]:
            wrong.append(f"{name}: «{text}» → {got} не понял: {spec['unknown']}")
    assert not wrong, "\n".join(wrong)


def test_every_pattern_has_phrases():
    from catalog import ALL_PATTERNS
    assert {p[0] for p in PHRASES} == set(ALL_PATTERNS)
