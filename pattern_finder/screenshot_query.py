"""Разовые эталоны для поиска по фотографии в чате."""

from catalog import SHAPE_PATTERNS
from fingerprint import kind_text, make_fingerprint, marked_items, visible_fvg_reference
from shape import SHAPE_POINTS, normalize, resample
from shapes_check import landmark_closes, validate
from similarity import reference_from_analysis, reference_to_json
from trendlines import FAMILY_BY_PATTERN, KIND_NAMES, drawn_span_has_swings, kind_from_drawing
from vision import UnreliableScreenshot, analyze_screenshot, extract_marks


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
        for name in SHAPE_PATTERNS:
            figure = validate(name, df, 0, len(df))
            if figure:
                segment = landmark_closes(df, figure)
                if len(segment) >= 8:
                    shape = normalize(resample(segment, SHAPE_POINTS)).tolist()
                    choices.append((name, {"name": name, "shape": shape, "shape_window": len(df)}))
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
    if not choices:
        raise UnreliableScreenshot("Свечи видны, но связать разметку с паттерном не удалось. "
                                   "Отметь границы одной фигуры или одну зону.")
    return choices
