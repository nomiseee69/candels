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

# ищутся по правилам на свечах (smc)
RULE_PATTERNS = ["FVG"]

SEARCHABLE = SHAPE_PATTERNS + LINE_PATTERNS + RULE_PATTERNS
