"""Мелочи для текстов бота."""


def plural(n, one, few, many):
    """Русские окончания: 1 случай, 3 случая, 11 случаев."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def _example(item):
    """«пример» или «пример №2», если примеров у паттерна несколько."""
    return f"пример №{item['example']}" if item.get("example") else "пример"


def quality_text(item):
    """Честная подпись к находке:
    фигуры из линий — число касаний (через 2 точки прямая проходит всегда, важны касания);
    паттерны по форме — схожесть с твоим скрином; SMC по правилам — без процентов
    (правило либо выполнено, либо нет, а детали — в описании)."""
    if isinstance(item.get("touches"), int):          # у EQH/EQL touches — список свечей, не число
        text = plural(item["touches"], "касание", "касания", "касаний")
        if item.get("similarity") is not None:
            text += f", похоже на твой {_example(item)} {item['similarity']:.0%}"
        return text
    if "kind" not in item and "zone" not in item and "level" not in item:
        if item.get("builtin"):
            return f"похоже на эталон {item['score']:.0%}"
        return f"похоже на твой {_example(item)} {item['score']:.0%}"
    if item.get("similarity") is not None:            # SMC: похожесть на отпечаток твоего примера
        return f"похоже на твой {_example(item)} {item['similarity']:.0%}"
    return ""
