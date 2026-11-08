"""Стресс-экзамен чтения скринов: те же случаи, что в screen_exam, но скрин сделан ПО-РАЗНОМУ —
как у разных пользователей: маленький с телефона, огромный с 4K-монитора, сжатый Telegram (JPEG),
мало свечей до паттерна, зона другого цвета (синяя, зелёная, красная, жёлтая), линии другого цвета.
Запуск:  python stress_exam.py            — все паттерны
         python stress_exam.py FVG BOS    — выбранные
"""

import os
import sys
import tempfile

from PIL import Image

import screen_exam as E
from catalog import ALL_PATTERNS, LINE_PATTERNS, SHAPE_PATTERNS

# вариант: (название, тема, объём, dpi, jpeg-качество, свечей до паттерна, цвет рисунка)
VARIANTS = [
    ("обычный", "dark", True, 100, None, 35, None),
    ("светлый без объёма", "light", False, 100, None, 35, None),
    ("маленький (телефон)", "dark", False, 60, None, 35, None),
    ("огромный (4K)", "light", True, 280, None, 35, None),
    ("сжатый Telegram", "dark", False, 140, 72, 35, None),
    ("мало свечей до", "dark", False, 100, None, 12, None),
    ("синий рисунок", "light", False, 100, None, 35, "#2962ff"),
    ("зелёный рисунок", "dark", True, 100, None, 35, "#4caf50"),
    ("красный рисунок", "light", True, 100, None, 35, "#f23645"),
    ("жёлтый рисунок", "dark", False, 100, None, 35, "#ffeb3b"),
]


def make_screen(window_df, drawing, folder, tag, theme, volume, dpi, jpeg, color):
    saved = dict(E.DRAW)
    if color:
        E.DRAW.update(zone=color, line=color, box=color)
    try:
        path = E.render(window_df, os.path.join(folder, f"{tag}.png"), theme, drawing, volume=volume, dpi=dpi)
    finally:
        E.DRAW.clear()
        E.DRAW.update(saved)
    if jpeg:
        out = path[:-4] + ".jpg"
        Image.open(path).convert("RGB").save(out, quality=jpeg)
        path = out
    return path


def cases_for(name, per_pattern, seeds, before):
    """[(окно свечей, правильный ответ, рисунок)] — как в screen_exam.run."""
    result = []
    if name in SHAPE_PATTERNS:
        for k in range(per_pattern * len(seeds)):
            window_df, key = E.planted(name, 100 + k)      # фигуре нужна вся форма — окно не режем
            result.append((window_df, {"key": key}, (None, None)))
        return result
    for seed in seeds:
        df = E.market(seed)
        for k, item in enumerate(E.cases(name, df, per_pattern)):
            start = int(item["start"])
            end = int(min(item.get("end", start + item.get("window", 1) - 1), len(df) - 1))
            lo, hi = max(0, start - before), min(len(df), end + 1 + E.AFTER)
            drawing = E._segment(item, lo)
            if name in LINE_PATTERNS and k % 2:
                drawing = (None, None)
            result.append((df.iloc[lo:hi].reset_index(drop=True), E._shift(item, lo), drawing))
    return result


def run(patterns=ALL_PATTERNS, per_pattern=3, seeds=(1, 2), say=print):
    folder = tempfile.mkdtemp()
    totals = {v[0]: [0, 0] for v in VARIANTS}
    table = {}
    for name in patterns:
        right = total = 0
        errors = []
        for variant in VARIANTS:
            label, theme, volume, dpi, jpeg, before, color = variant
            for i, (window_df, item, drawing) in enumerate(cases_for(name, per_pattern, seeds, before)):
                tag = f"{abs(hash((name, label, i)))}"
                path = make_screen(window_df, drawing, folder, tag, theme, volume, dpi, jpeg, color)
                try:
                    ok, seen = E.check(name, item, window_df, path)
                except Exception as error:
                    ok, seen = False, f"ошибка: {error!r}"
                right += ok
                total += 1
                totals[label][0] += ok
                totals[label][1] += 1
                if not ok and len(errors) < 4:
                    errors.append(f"[{label}] {seen}")
        table[name] = (right, total, errors)
        say(f"{name:18s} {right:3d}/{total:<3d} " + "; ".join(errors))
    say("\nПо видам скринов:")
    for label, (r, t) in totals.items():
        say(f"  {label:22s} {r}/{t}")
    return table, totals


if __name__ == "__main__":
    run(sys.argv[1:] or ALL_PATTERNS)
