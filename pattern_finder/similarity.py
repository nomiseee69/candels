"""«Похоже на твой пример»: насколько найденная фигура похожа на скрин пользователя.

Масштаб скрина и биржевого графика разный (пиксели против цены, 3 часа против 3 недель),
поэтому сравниваем не сами числа, а ФОРМУ — каждую фигуру вписываем в квадрат 1×1:
  время от начала до конца фигуры → 0..1,
  цена от самой нижней до самой верхней точки линий → 0..1.
После этого у двух фигур можно сравнить:
  1) геометрию линий — где начинаются и заканчиваются верхняя и нижняя линия
     (крутизна, сужение, направление);
  2) путь цены внутри фигуры — как цена ходила между линиями (корреляция).
Итог: 60% геометрия + 40% путь цены, от 0 до 1.
"""

import numpy as np

from shape import normalize, resample
from vision import analyze_screenshot

PATH_POINTS = 50
GEOMETRY_WEIGHT = 0.6
SIMILAR_ENOUGH = 0.6          # настройка «только похожие на мой пример»: от 60%


def _value(line, x):
    x0, y0, x1, y1 = line
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0


def _unit_box(upper, lower, left, right):
    """Значения линий на краях фигуры, вписанные в квадрат 1×1: (u0, u1, l0, l1)."""
    values = [_value(upper, left), _value(upper, right), _value(lower, left), _value(lower, right)]
    low, high = min(values), max(values)
    if high - low <= 0:
        return None
    return tuple((v - low) / (high - low) for v in values)


def reference_from_screenshot(path):
    """Эталон с рисунка пользователя: геометрия двух линий и путь цены между ними.
    Нет двух линий на скрине — None (сравнивать не с чем)."""
    try:
        analysis = analyze_screenshot(path)
    except (OSError, ValueError):
        return None
    return reference_from_analysis(analysis)


def reference_from_analysis(analysis):
    """Геометрия и путь цены из уже разобранного изображения."""
    if not analysis.ok or len(analysis.lines) != 2:
        return None
    h = analysis.height
    left = min(ln.x1 for ln in analysis.lines)
    right = max(ln.x2 for ln in analysis.lines)
    if right - left < 10:
        return None
    # на картинке y растёт вниз — переворачиваем, чтобы «вверх» было «дороже»
    lines = [(ln.x1, h - 1 - ln.y1, ln.x2, h - 1 - ln.y2) for ln in analysis.lines]
    middle = (left + right) / 2
    upper, lower = sorted(lines, key=lambda ln: _value(ln, middle), reverse=True)
    geometry = _unit_box(upper, lower, left, right)
    if geometry is None:
        return None
    inside = (analysis.xs >= left) & (analysis.xs <= right)
    path = None
    if inside.sum() >= 10:
        path = normalize(resample(analysis.prices[inside], PATH_POINTS))
    return {"geometry": geometry, "path": path}


def figure_shape(df, item):
    """То же для фигуры на рынке: геометрия линий и путь цены (close) внутри неё."""
    left = min(item["upper"][0], item["lower"][0])
    right = item["upper"][2]
    geometry = _unit_box(item["upper"], item["lower"], left, right)
    closes = df["close"].to_numpy()[int(left):int(right) + 1]
    path = normalize(resample(closes, PATH_POINTS)) if len(closes) >= 5 else None
    return {"geometry": geometry, "path": path}


def compare(reference, figure):
    """Похожесть двух фигур от 0 до 1."""
    if reference is None or figure["geometry"] is None:
        return None
    diff = np.mean(np.abs(np.subtract(reference["geometry"], figure["geometry"])))
    geometry = max(0.0, 1.0 - 2.0 * float(diff))        # средняя разница 0.25 → 50%
    if reference["path"] is None or figure["path"] is None:
        return geometry
    corr = np.corrcoef(reference["path"], figure["path"])[0, 1]
    path = max(0.0, float(corr)) if np.isfinite(corr) else 0.0
    return GEOMETRY_WEIGHT * geometry + (1 - GEOMETRY_WEIGHT) * path


def best_match(df, item, references):
    """Самый похожий из примеров пользователя: (похожесть, номер примера) или (None, None)."""
    figure = figure_shape(df, item)
    scored = [(s, i) for i, s in enumerate(compare(ref, figure) for ref in references) if s is not None]
    return max(scored) if scored else (None, None)


def best_similarity(df, item, references):
    """Похожесть на самый близкий из примеров пользователя (их может быть несколько)."""
    return best_match(df, item, references)[0]


def reference_to_json(reference):
    """Эталон → обычные списки, чтобы сохранить в users.json."""
    path = reference["path"]
    return {"geometry": list(map(float, reference["geometry"])),
            "path": None if path is None else [float(v) for v in path]}


def example_from_item(df, item):
    """Находка с рынка → эталон для нового примера (кнопка «💾 Сохранить как пример»).
    Фигура из линий — геометрия и путь цены; паттерн по форме — форма цены.
    У SMC-паттернов эталона нет (их ищем по правилам) — None."""
    if "upper" in item:
        return {"reference": reference_to_json(figure_shape(df, item)), "kind": item.get("kind")}
    if "kind" not in item and "zone" not in item and "level" not in item:
        closes = df["close"].to_numpy()[item["start"]:item["start"] + item["window"]]
        return {"shape": [float(v) for v in normalize(resample(closes, PATH_POINTS))]}
    return None
