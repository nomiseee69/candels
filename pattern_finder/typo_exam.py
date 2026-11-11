"""Экзамен «понимает ли бот описание с опечатками» — без нейросети, одним словарём.

Берём живые трейдерские фразы по всем 17 паттернам (tests/test_phrases.py), портим их так, как
портят люди, когда печатают быстро, и проверяем: понял ли бот то же самое, что и без опечаток.
Как портим:
  • опечатка    — соседние буквы поменялись местами, буква пропала, удвоилась или заменилась
                  на соседнюю клавишу (в одном или двух длинных словах);
  • раскладка   — фраза набрана в английской раскладке («adu» вместо «фвг»);
  • без пробелов и знаков — «свежий фвг на 4ч рядом с ценой» без запятых;
  • КАПС.
Отдельно — названия паттернов с опечатками и на сленге («ордр блок», «дабл топ», «фвж»).
Запуск:  python typo_exam.py
"""

import random
import sys

sys.path.insert(0, "tests")

import language

NEAR = {  # соседние клавиши русской раскладки
    "й": "цф", "ц": "йук", "у": "цкв", "к": "уеа", "е": "кнп", "н": "егр", "г": "нш", "ш": "гщл", "щ": "шзд",
    "з": "щх", "ф": "йыя", "ы": "фвч", "в": "ыау", "а": "впк", "п": "аре", "р": "пон", "о": "рлг", "л": "одш",
    "д": "лжщ", "ж": "дэ", "я": "фч", "ч": "яыс", "с": "чмв", "м": "сиа", "и": "мтп", "т": "иьр", "ь": "тбо",
    "б": "ью",
}
TO_LATIN = {**str.maketrans("йцукенгшщзхъфывапролджэячсмитьбюё", "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"), ord(","): "?", ord("."): "/"}

NAMES = {   # как люди называют паттерны: сленг, английский, опечатки
    "FVG": ["фвг", "фвж", "фгв", "fvg", "имбаланс", "имба", "имбаланас", "fair value gap", "гэп", "имбалнс", "adu"],
    "Order Block": ["ордер блок", "ордерблок", "ордр блок", "order block", "order blok", "ob", "об", "ордерблоки",
                    "оредр блок", "jhlth ,kjr"],
    "Breaker Block": ["брейкер", "брекер блок", "breaker", "брейкр", "брэйкер блок", "бреикер"],
    "Mitigation Block": ["митигейшн", "митигэйшен блок", "mitigation", "митигейшен", "митигация", "митигешн блок"],
    "BOS": ["бос", "bos", "слом структуры", "слом стурктуры", "пробой структуры", "break of structure", "слом структры"],
    "CHoCH": ["чоч", "choch", "чох", "смена характера", "смена хаарктера", "change of character", "чейндж оф характер"],
    "EQH / EQL": ["равные хаи", "равные лои", "eqh", "eql", "равные максимумы", "равные минимумы", "ровные хаи",
                  "equal highs", "equal lows", "двойные лои с ликвидностью"],
    "Range": ["боковик", "флэт", "рендж", "range", "проторговка", "консолидация", "накопление", "бокавик",
              "диапазон", "флет", "проторговкка"],
    "Двойное дно": ["двойное дно", "двоное дно", "double bottom", "дабл боттом", "двойноe дно", "дв дно", "двойное днр"],
    "Двойная вершина": ["двойная вершина", "двойной топ", "double top", "дабл топ", "двойная вершниа", "двойная вершна"],
    "Голова и плечи": ["голова и плечи", "гип", "голова плечи", "head and shoulders", "голова и плчи", "галова и плечи"],
    "Перевёрнутая ГиП": ["перевёрнутая гип", "перевернутая голова и плечи", "обратная гип", "inverse head and shoulders",
                         "перевернутые голова плечи", "перевернутая галова и плечи", "обратная голова плечи"],
    "Треугольник": ["треугольник", "триугольник", "triangle", "треуголник", "треугольничек", "трегольник"],
    "Клин": ["клин", "wedge", "клинышек", "клин падающий"],
    "Флаг": ["флаг", "flag", "флажок", "флга"],
    "Вымпел": ["вымпел", "вымпл", "pennant", "вымпeл", "выпмел"],
    "Канал": ["канал", "channel", "каанл", "кнал", "параллельный канал"],
}


def typo(word, rng):
    """Одна опечатка в слове (первую букву не трогаем — люди её обычно набирают верно)."""
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "double", "near"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice(NEAR.get(word[i], word[i])) + word[i + 1:]


def spoil(text, rng, words=1):
    parts = text.split(" ")
    long = [i for i, w in enumerate(parts) if len(w.strip(",.")) >= 6 and w.strip(",.").isalpha()]
    for i in rng.sample(long, min(words, len(long))):
        tail = "," if parts[i].endswith(",") else ""
        parts[i] = typo(parts[i].rstrip(","), rng) + tail
    return " ".join(parts)


def understood(text, name):
    spec = language.parse(text, name)
    return (sorted(spec["require"]), sorted(spec["prefer"]), sorted(spec["forbid"]), spec["timeframes"],
            spec["direction"], spec["min_touches"])


def run(say=print, rounds=6):
    from test_phrases import PHRASES
    kinds = {"одна опечатка": lambda t, r: spoil(t, r, 1), "две опечатки": lambda t, r: spoil(t, r, 2),
             "английская раскладка": lambda t, r: t.lower().translate(TO_LATIN),
             "без запятых": lambda t, r: t.replace(",", ""), "КАПС": lambda t, r: t.upper()}
    table = {}
    for label, change in kinds.items():
        right = total = 0
        errors = []
        for k in range(rounds if "опечат" in label else 1):
            rng = random.Random(1000 + k)
            for name, text, *_ in PHRASES:
                want = understood(text, name)
                bad = change(text, rng)
                ok = understood(bad, name) == want
                right += ok
                total += 1
                if not ok and len(errors) < 4:
                    errors.append(f"«{bad}»")
        table[label] = (right, total)
        say(f"{label:22s} {right:3d}/{total:<3d} " + "; ".join(errors))
    right = total = 0
    errors = []
    for name, variants in NAMES.items():
        for text in variants:
            ok = language.detect_pattern(text) == name
            right += ok
            total += 1
            if not ok:
                errors.append(f"«{text}»→{language.detect_pattern(text)}")
    table["названия паттернов"] = (right, total)
    say(f"{'названия паттернов':22s} {right:3d}/{total:<3d} " + "; ".join(errors[:12]))
    return table


if __name__ == "__main__":
    run()
