"""Мелочи для текстов бота."""


def plural(n, one, few, many):
    """Русские окончания: 1 случай, 3 случая, 11 случаев."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def direction_arrow(item):
    """↑ / ↓ для паттернов с направлением, пусто — для фигур без направления."""
    kind = str(item.get("kind", ""))
    if kind.startswith("bullish") or item.get("direction") == "up":
        return " ↑"
    if kind.startswith("bearish") or item.get("direction") == "down":
        return " ↓"
    return ""


def where_text(distance, inside="цена в зоне"):
    """Где цена относительно зоны / уровня — словами, без процентов."""
    if distance == 0:
        return inside
    return "цена рядом" if distance <= 0.01 else "цена недалеко" if distance <= 0.05 else "цена далеко"


def finding_name(item):
    """Как находка выглядит в выдаче: только паттерн и направление — «Order Block ↑».
    Никаких процентов и статистики: сила находки решает, ЧТО прислать, а не печатается."""
    return f"{item['name']}{direction_arrow(item)}"


def quality_text(item):
    """Раньше — «похоже на N%», «шанс реакции N%». Числа в выдаче больше не пишем."""
    return ""


def similarity_words(value):
    """Похожесть на пример словами (для подписи к сохранённому примеру и т.п.)."""
    if value is None:
        return ""
    if value >= 0.8:
        return "очень похож на твой пример"
    if value >= 0.6:
        return "похож на твой пример"
    return "отдалённо похож на твой пример"
