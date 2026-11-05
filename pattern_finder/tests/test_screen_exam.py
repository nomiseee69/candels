"""Экзамен чтения скриншотов по ВСЕМ паттернам (малый вариант; полный —  python screen_exam.py)."""

import screen_exam
from catalog import ALL_PATTERNS


def test_every_pattern_is_understood_from_screenshots():
    table = screen_exam.run(ALL_PATTERNS, per_pattern=2, seeds=(1, 2), say=lambda text: None)
    assert set(table) == set(ALL_PATTERNS)
    right = sum(r for r, _, _ in table.values())
    total = sum(t for _, t, _ in table.values())
    assert total >= 50 and right >= 0.85 * total, table
    for name, (r, t, errors) in table.items():
        assert t == 0 or r >= t / 2, (name, errors)          # ни один паттерн не проваливается
