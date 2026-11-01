"""Мелочи для текстов бота."""


def plural(n, one, few, many):
    """Русские окончания: 1 случай, 3 случая, 11 случаев."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def quality_text(item):
    """Честная подпись к находке:
    фигуры из линий — число касаний (через 2 точки прямая проходит всегда, важны касания);
    паттерны по форме — схожесть с твоим скрином; SMC по правилам — без процентов
    (правило либо выполнено, либо нет, а детали — в описании)."""
    if isinstance(item.get("touches"), int):          # у EQH/EQL touches — список свечей, не число
        return plural(item["touches"], "касание", "касания", "касаний")
    if "kind" not in item and "zone" not in item and "level" not in item:
        return f"схожесть {item['score']:.0%}"
    return ""
