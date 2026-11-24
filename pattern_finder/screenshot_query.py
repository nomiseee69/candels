"""Разовые эталоны для поиска по фотографии в чате."""

from catalog import SHAPE_PATTERNS
from fingerprint import kind_text, make_fingerprint, marked_items, visible_fvg_reference
from shape import SHAPE_POINTS, normalize, resample
from shapes_check import landmark_closes, marked_double_reversal, marked_head_shoulders, validate
from similarity import reference_from_analysis, reference_to_json
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, drawn_span_has_swings, kind_from_drawing
from vision import UnreliableScreenshot, analyze_screenshot, extract_marks


def _shape_choice(name, df, figure, window):
    segment = landmark_closes(df, figure)
    if len(segment) >= 8:
        shape = normalize(resample(segment, SHAPE_POINTS)).tolist()
        reference = {"name": name, "shape": shape, "shape_window": window}
        if "neck_template" in figure:
            reference["neck_template"] = figure["neck_template"]
        return name, reference
    return None


def _marked_reversals(df, lines):
    if not 1 <= len(lines) <= 5:
        return
    for name in SHAPE_PATTERNS:
        if name in ("Двойное дно", "Двойная вершина"):
            match = marked_double_reversal(name, df, lines)
        else:
            match = marked_head_shoulders(name, df, lines)
        if match is not None:
            choice = _shape_choice(name, df, *match)
            if choice:
                yield choice


def screenshot_choices(path):
    """Возвращает пары (что увидел пользователь, аргументы существующего сканера).

    Для зон и уровней требуем совпадения разметки с отдельной находкой правила.
    Сам по себе нарисованный прямоугольник ещё не указывает тип и направление.
    """
    df, zones, lines = extract_marks(path)
    if df is None or len(df) < 12:
        raise UnreliableScreenshot("Не удалось восстановить достаточно свечей. Увеличь участок с фигурой.")
    choices = []
    if len(zones) > 3:
        raise UnreliableScreenshot("На графике слишком много зон. Оставь нужную зону и соседние свечи.")
    if zones:
        families = ("FVG", "Order Block", "Breaker Block", "Mitigation Block", "Range")
        zones = sorted(zones, key=lambda z: (z["start"], -z["top"]))
    elif lines:
        families = ("BOS", "CHoCH", "EQH / EQL") if any(ln["flat"] for ln in lines) else ()
        if len(lines) == 2 and drawn_span_has_swings(df, lines):
            analysis = analyze_screenshot(path)
            kind = kind_from_drawing(analysis.lines)
            reference = reference_from_analysis(analysis, visible_span=True)
            if reference and kind:
                name = next((n for n, kinds in FAMILY_BY_PATTERN.items() if kind in kinds), None)
                if name:
                    choices.append((KIND_NAMES[kind], {"name": name, "kind": kind,
                                                       "reference": reference_to_json(reference)}))
        if 2 <= len(lines) <= 4:
            families += ("Флаг", "Вымпел")
    else:
        families = ()
    if not zones and not lines:
        for name in SHAPE_PATTERNS:
            window = len(df)
            figure = validate(name, df, 0, window)
            if figure:
                choice = _shape_choice(name, df, figure, window)
                if choice:
                    choices.append(choice)
    for name in families:
        for pos, (overlap, item) in enumerate(marked_items(name, df, zones, lines)):
            if overlap < 0.70:
                continue
            label = kind_text({**item, "name": name})
            if len(zones) > 1:
                label += f" · зона {pos + 1} слева"
            fp = make_fingerprint(name, df, item)
            if name == "FVG":
                fp = visible_fvg_reference(fp, df, item)
                if fp is None:
                    continue
            choices.append((label, {"name": name, "kind": item.get("kind"), "direction": item.get("direction"),
                                    "fingerprint": fp}))
    # Неподтверждённая зона не отменяет разворот, привязанный к разметке
    # и подтверждённый свечами. Без такой привязки сохраняем отказ.
    # Подтверждённая пара границ использует обе линии. Те же линии не
    # следует считать независимой разметкой шеи соседней фигуры.
    has_boundary_pair = any("reference" in query for _, query in choices)
    if lines and (not zones or not choices) and not has_boundary_pair:
        choices.extend(_marked_reversals(df, [line for line in lines if not line.get("zone")]))
    if not choices:
        raise UnreliableScreenshot("Свечи видны, но связать разметку с паттерном не удалось. "
                                   "Отметь границы одной фигуры или одну зону.")
    return choices
