"""Все паттерны, которые можно выбрать в боте, и как их искать."""

TA_PATTERNS = [
    "Двойное дно", "Двойная вершина", "Голова и плечи", "Перевёрнутая ГиП",
    "Треугольник", "Клин", "Флаг", "Вымпел", "Канал",
]

SMC_PATTERNS = [
    "FVG", "Order Block", "Breaker Block", "Mitigation Block",
    "BOS", "CHoCH", "EQH / EQL", "Range",
]

ALL_PATTERNS = TA_PATTERNS + SMC_PATTERNS

# ищутся по форме с картинки пользователя (matcher)
SHAPE_PATTERNS = ["Двойное дно", "Двойная вершина", "Голова и плечи", "Перевёрнутая ГиП"]

# ищутся по линиям тренда через экстремумы (trendlines)
LINE_PATTERNS = ["Треугольник", "Клин", "Канал"]

# флагшток + короткий канал (флаг) или маленький треугольник (вымпел) — flags.py
FLAG_PATTERNS = ["Флаг", "Вымпел"]
FLAG_SHAPE = {"Флаг": "flag", "Вымпел": "pennant"}

# ищутся по правилам на свечах (smc, structure, blocks, ranges)
RULE_PATTERNS = [
    "FVG", "BOS", "CHoCH", "EQH / EQL", "Order Block", "Breaker Block", "Mitigation Block", "Range",
]

SEARCHABLE = SHAPE_PATTERNS + LINE_PATTERNS + FLAG_PATTERNS + RULE_PATTERNS
