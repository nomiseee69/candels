"""Встроенный «переводчик» описаний: трейдер пишет словами, каким должен быть ЕГО паттерн,
а бот превращает текст в чёткие условия и ищет на рынке именно такой паттерн.

Без интернета и внешних нейросетей. Понимание в три слоя:
  1) СЛОВАРЬ ТРЕЙДЕРА — для каждого понятия (снятие ликвидности, объём, поглощение, ключевой
     слом, смена структуры, свежая зона, дисконт…) сотни вариантов: корни слов, сленг RU/EN
     («свип», «стопы собрал», «вынос», «имба», «fresh», «unmitigated»);
  2) НЕЧЁТКОЕ СРАВНЕНИЕ ФРАЗ — для каждого понятия есть «банк» живых формулировок трейдеров.
     Слова приводятся к основе (снял / снятие / снимает → «сня»), опечатки прощаются, и фраза
     пользователя сравнивается с банком: «BOS у последней свечи, которую если пересечь — меняется
     структура» узнаётся как «ключевой слом (смена структуры)», хотя таких слов в словаре нет;
  3) СМЫСЛ ВОКРУГ — «обязательно / только / должен» → обязательно, «лучше / желательно» →
     желательно, «без / не / кроме» → нельзя; таймфреймы, направление, числа, связки паттернов.

Чего бот не понял — честно говорит. Условия проверяются на каждой находке (критерии эксперта
strength.py + поля паттерна) и вместе с примерами-скринами и оценками 👍/👎 (feedback.py)
решают, ЧТО бот найдёт.
"""

import difflib
import re

import numpy as np

GROUPS = {   # паттерн → группа понятий (какие условия к нему вообще относятся)
    "FVG": "zone", "Order Block": "zone", "Breaker Block": "zone", "Mitigation Block": "zone",
    "BOS": "structure", "CHoCH": "structure", "EQH / EQL": "eq", "Range": "range",
    "Двойное дно": "shape", "Двойная вершина": "shape", "Голова и плечи": "shape", "Перевёрнутая ГиП": "shape",
    "Треугольник": "figure", "Клин": "figure", "Канал": "figure", "Флаг": "flag", "Вымпел": "flag",
}
ALL = ("zone", "structure", "eq", "range", "shape", "figure", "flag")

# понятие: (как назвать в ответе, регулярки, банк фраз для нечёткого сравнения, группы)
CONCEPTS = {
    "liquidity": ("снятие ликвидности (прошлый свинг)",
                  [r"ликвид", r"\bликв", r"сбор\w*\s+(ликв|стоп)", r"ложн\w*\s+пробо", r"\bsweep", r"свип", r"сн(ял|яла|яли|имает|ятие|ять)\s+(стоп|хай|лой|ликв|свинг|макс|мин)",
                   r"забрал\w*\s+(стоп|хай|лой|ликв)", r"собрал\w*\s+(стоп|ликв)", r"прострел", r"выбил\w*\s+стоп",
                   r"\bstop\s*hunt", r"\bliquidity", r"\bssl\b", r"\bbsl\b", r"\bгрэб", r"\bgrab", r"выколол",
                   r"проколол\w*\s+(хай|лой|минимум|максимум|свинг|уровен)"],
                  ["снял стопы", "собрал ликвидность", "забрал хаи", "забрал лои", "прокол прошлого минимума",
                   "вынес стопы", "ложный пробой уровня", "сходил за минимум", "сходил за максимум",
                   "прошёл за свинг и вернулся", "охота за стопами"],
                  ("zone", "structure", "range")),
    "volume": ("повышенный объём",
               [r"объ[её]м", r"\bvolume", r"\bvol\b", r"вол[ьи]ум", r"\bвыс\w*\s+активн"],
               ["большой объём", "всплеск объёма", "объём выше среднего", "крупный игрок зашёл", "аномальный объём"],
               ("zone", "structure", "shape", "flag", "range", "figure")),
    "engulf": ("поглощение: следующая свеча закрылась за тенью",
               [r"поглощ", r"\bengulf", r"закрыл\w*\s+(ниже|выше|за)\s+(тен|фитил|лоя|хая|low|high|минимум|максимум)",
                r"противоположн\w*\s+свеч", r"следующ\w*\s+свеч", r"перекрыва\w*\s+(тело|тень|свеч|блок)", r"перекрыл\w*\s+свеч", r"перекрыл\w*\s+(тело|тень)"],
               ["следующая свеча закрылась ниже тени", "следующая свеча закрылась выше тени",
                "сразу противоположная свеча", "свеча перекрыла блок", "закрытие за минимумом блока"],
               ("zone",)),
    "impulse": ("резкий импульс от зоны",
                [r"импульс", r"\bdisplacement", r"резк\w*\s+(движ|уход|выход|свеч|рост|паден|вынос|отскок)",
                 r"сильн\w*\s+(движ|свеч|импульс|вынос|отскок)", r"вынос", r"больш\w*\s+свеч", r"улет", r"выстрел",
                 r"пампан|дампан|\bpump|\bdump"],
                ["цена резко ушла от зоны", "мощное движение", "быстро ушли", "сильный отскок", "резкий рост",
                 "резкое падение"],
                ("zone", "structure", "flag")),
    "imbalance": ("FVG / имбаланс внутри",
                  [r"\bfvg", r"имбаланс", r"\bimbalance", r"\bимба\b", r"гэп", r"\bgap\b", r"дыр[аыу]", r"разрыв",
                   r"\bifvg", r"неэффективн"],
                  ["оставил разрыв", "незаполненная дыра", "неэффективность цены"],
                  ("zone",)),
    "structure": ("слом структуры (BOS) от зоны",
                  [r"\bbos\b", r"слом\w*\s+структ", r"сломал\w*\s+структ", r"пробил\w*\s+(хай|лой|структ|свинг)",
                   r"\bchoch", r"\bmss\b", r"break\s+of\s+structure"],
                  ["после слома структуры", "пробил предыдущий максимум", "пробил предыдущий минимум"],
                  ("zone",)),
    "key": ("ключевой слом — меняет структуру / тренд (последний значимый)",
            [r"ключев", r"главн\w*\s+(слом|bos|уров|свинг)", r"смен\w*\s+(тренд|структур|контекст|характер|направлен)",
             r"мен(яет|яется|ять)\s+(тренд|структур|контекст|направлен)", r"последн\w*\s+(слом|bos|свинг|свеч|значим|хай|лой)",
             r"только\s+последн", r"защищ\w*", r"\bstrong\s+(high|low)", r"\bprotected", r"сильн\w*\s+(хай|лой|минимум|максимум)",
             r"если\s+(пересеч|пробь|пробит|сломать|закро)", r"разворот\w*\s+(тренд|структур)", r"контекст"],
            ["должен определяться только последний bos", "свеча которую если пересечь меняется структура рынка",
             "уровень после пробоя которого меняется тренд", "только тот слом который меняет структуру",
             "смена контекста рынка", "ключевой уровень структуры", "последний значимый свинг",
             "если цена его пробьёт тренд сменится", "уровень слома тренда", "защищённый минимум",
             "защищённый максимум", "точка смены тренда", "без лишних сломов только главный"],
            ("structure",)),
    "major": ("крупная (внешняя) структура — большой свинг",
              [r"внешн\w*\s+структ", r"\bexternal", r"\bmajor", r"крупн\w*\s+(свинг|структ|слом|движ)",
               r"больш\w*\s+(свинг|слом|движ)", r"старш\w*\s+(структ|свинг)", r"\bswing\s+structure"],
              ["крупный свинг", "внешняя структура рынка", "значимый большой слом", "не мелкий слом"],
              ("structure",)),
    "internal": ("мелкая (внутренняя) структура",
                 [r"внутренн\w*\s+структ", r"\binternal", r"\bminor", r"мелк\w*\s+(свинг|структ|слом)",
                  r"локальн\w*\s+(свинг|слом)"],
                 ["внутренняя структура", "мелкие сломы", "локальный слом"],
                 ("structure",)),
    "discount": ("в дисконте (бычий) / в премиуме (медвежий)",
                 [r"дисконт", r"discount", r"премиум", r"premium", r"дешев", r"дорог\w*\s+зон", r"скидк",
                  r"нижн\w*\s+половин", r"верхн\w*\s+половин", r"\bfib", r"фибо", r"\b0[.,]5\b", r"\bequilibrium"],
                 ["ниже середины диапазона", "выше середины диапазона", "в нижней части движения",
                  "в верхней части движения", "по фибоначчи ниже половины"],
                 ("zone",)),
    "trend": ("по тренду",
              [r"по\s+тренд", r"тренд", r"по\s+направлени", r"with\s+trend", r"по\s+структур", r"по\s+ходу\s+движ"],
              ["в сторону основного движения", "по направлению рынка", "трендовый"],
              ("zone", "structure", "flag")),
    "fresh": ("свежая — цена ещё не возвращалась",
              [r"нетронут", r"не\s*тронут", r"свеж", r"не\s*тестир", r"не\s*протестир", r"перв\w*\s+(ретест|касани|тест)",
               r"не\s*митиг", r"\bfresh", r"\bunmitigated", r"не\s*отработ", r"не\s*задет", r"цел[ао]я?\b",
               r"без\s+(касан|ретест|тест|возврат)", r"не\s*перекрыт", r"не\s*закрыт", r"не\s*отработ", r"цена\s+не\s+(возвращ|приход|заход)", r"не\s*заполн", r"чист\w*\s+зон"],
              ["цена ещё не возвращалась в зону", "зона не отработана", "первое касание зоны", "не было ретеста"],
              ("zone",)),
    "retest": ("уже был ретест / цена вернулась",
               [r"ретест", r"\bretest", r"верн(улась|улись|ется)", r"протестир\w*\s+(зон|уров|блок)", r"тестир\w*\s+(зон|уров)"],
               ["цена вернулась к уровню", "был ретест уровня", "проверили уровень сверху"],
               ("zone", "structure", "range", "figure", "flag", "shape")),
    "near_price": ("рядом с текущей ценой",
                   [r"рядом\s+с\s+цен", r"близко\s+к\s+цен", r"возле\s+цен", r"актуальн", r"около\s+цен", r"цена\s+подош",
                    r"подход\w*\s+к\s+(зон|уровн)", r"\bсейчас\b", r"в\s+зоне\s+сейчас"],
                   ["цена подходит к зоне", "зона рядом с ценой", "актуальная зона"],
                   ("zone", "eq")),
    "recent": ("свежая по времени — сложилась недавно",
               [r"недавн", r"последн\w*\s+(свеч|дн|час)", r"только\s+что", r"\brecent", r"нов\w*\s+(зон|блок|fvg|паттерн)"],
               ["сложилась только что", "на последних свечах"],
               ("zone", "structure", "eq", "figure", "flag", "shape", "range")),
    "big": ("крупная зона / фигура",
            [r"крупн", r"больш(ой|ая|ое|ие|ую)\s+(зон|блок|fvg|ob|дыр|фигур|канал|боков|range|треуг)", r"широк",
             r"сильн\w*\s+зон", r"мощн\w*\s+(зон|блок|уров|фигур)", r"заметн\w*\s+(зон|фигур)"],
            ["хорошо видна на графике", "большая зона"],
            ("zone", "shape", "figure", "range")),
    "small": ("узкая зона / фигура",
              [r"мелк", r"узк", r"маленьк", r"небольш", r"тонк\w*\s+зон"],
              ["маленькая зона"],
              ("zone", "shape", "figure", "range")),
    "partial": ("частично заполнен",
                [r"частичн", r"наполовину", r"до\s+середин", r"\b50\s*%", r"\bce\b", r"consequent"],
                ["цена зашла в половину дыры"],
                ("zone",)),
    "body_close": ("закрытие ТЕЛОМ свечи за уровнем",
                   [r"тел(ом|о свеч)", r"закрыт\w*\s+(тел|свеч)", r"не\s+тен\w*", r"не\s+фитил", r"закрепил"],
                   ["свеча закрылась за уровнем", "закрепление за уровнем", "не просто прокол тенью"],
                   ("structure", "range", "figure")),
    "untapped": ("ликвидность ещё не снята",
                 [r"не\s*снят", r"не\s*собран", r"не\s*забран", r"нетронут", r"ещ[её]\s+не\s+сня", r"висит"],
                 ["уровень ещё не снимали", "стопы ещё стоят"],
                 ("eq",)),
    "swept": ("ликвидность уже сняли",
              [r"(уже\s+)?снял[аи]?\b(?!\s+ли)", r"(уже\s+)?снят[аы]?\b", r"собрал[аи]?\b", r"прострелил"],
              ["стопы уже собрали"],
              ("eq",)),
    "tight": ("ровный уровень",
              [r"ровн", r"точн\w*\s+(уров|равн)", r"одн\w*\s+уровн", r"идеальн\w*\s+равн"],
              ["вершины на одном уровне", "минимумы на одной цене"],
              ("eq", "range")),
    "many_touches": ("много касаний",
                     [r"много\s+касан", r"многократн", r"несколько\s+раз", r"часто\s+(касал|тестир)"],
                     ["цена много раз отталкивалась"],
                     ("eq", "range", "figure", "flag")),
    "confirmed": ("пробой подтверждён (шея / граница пробита)",
                  [r"шея\s+пробит", r"пробо\w*\s+ше", r"подтвержд", r"пробил\w*\s+шею", r"закрыл\w*\s+за\s+ше",
                   r"пробит\w*\s+(границ|лини|уров)", r"вышл\w*\s+из\s+(фигур|канал|треуг|боков|флаг)", r"\bbreakout",
                   r"пробой"],
                  ["цена вышла из фигуры", "пробой границы", "закрепились за линией"],
                  ("shape", "figure", "flag", "range")),
    "symmetry": ("симметричная фигура",
                 [r"симметр", r"одинаков\w*\s+(дн|плеч|вершин)", r"плечи\s+на\s+одном", r"на\s+одном\s+уровн",
                  r"ровн\w*\s+(дн|вершин|плеч)"],
                 ["оба дна на одном уровне", "плечи одинаковые"],
                 ("shape",)),
    "compression": ("сужение (линии сходятся)",
                    [r"сужен", r"(?<![а-яё])сход(ятся|ящ)", r"сжат", r"\bsqueeze", r"\bcompression", r"схожд"],
                    ["цена сжимается", "линии сходятся"],
                    ("figure", "flag")),
    "near_liq": ("рядом с равными хаями / лоями (ликвидность)",
                 [r"(у|возле|рядом\s+с|около|под|над)\s+(равн|eqh|eql|ликвид)", r"\beqh\b", r"\beql\b",
                  r"равн\w*\s+(хаи|лои|хай|лой|вершин|дн|максимум|минимум)"],
                 ["рядом с равными максимумами", "под равными минимумами"],
                 ("zone",)),
    "inside_ob": ("FVG внутри Order Block",
                  [r"(fvg|имбаланс|гэп|имба)\s+(внутри|в)\s+(ob|ордер|блок)", r"(внутри|в)\s+(ob|ордер\s*блок)",
                   r"(ob|ордер\s*блок\w*)\s+с\s+(fvg|имбаланс|имба)", r"совпад\w*\s+с\s+(ob|ордер|блок)"],
                  ["дыра совпадает с ордерблоком", "имбаланс в зоне блока"],
                  ("zone",)),
}

PREFER = r"(желательн|лучше|предпочт|приоритет|по\s*возможност|хорошо\s+бы|плюсом|бонус|в\s+идеале|идеально|приветств)"
REQUIRE = r"(обязательн|только|должн|нуж(ен|на|но|ны)|строго|всегда|точно|непременно|важно|главное|необходим)"
NEGATION_WORDS = ("не", "без", "нет", "ни", "кроме", "исключ", "никак", "not", "no", "without", "против")
TIMEFRAMES = {"1h": [r"\b1\s*[hч]\b", r"\b1\s*час", r"\bчасов(ик|ой|ом|ке|ка)\b", r"\bh1\b", r"\b60\s*м"],
              "4h": [r"\b4\s*[hч]\b", r"\b4\s*час", r"\bh4\b", r"четыр\w*\s*час", r"четыр[её]хчас"],
              "12h": [r"\b12\s*[hч]\b", r"\b12\s*час", r"\bh12\b", r"двенадцати"],
              "1d": [r"\b1\s*[dд]\b", r"\bд1\b", r"\bd1\b", r"дневк", r"дневн", r"\bдень\b", r"\bdaily", r"\bдэйли"],
              "1w": [r"\b1\s*[wн]\b", r"\bw1\b", r"недел", r"\bweekly", r"\bвикли"]}
OLDER = r"(старш|высок|больш|крупн)\w*\s+(тф|таймфрейм|tf|интервал)|\bhtf\b"
YOUNGER = r"(младш|мелк|низк|маленьк)\w*\s+(тф|таймфрейм|tf|интервал)|\bltf\b"
BULLISH = [r"бычь?", r"\bлонг", r"\blong\b", r"покуп", r"вверх", r"\bbull", r"\bbuy", r"\bрост\b", r"поддержк",
           r"\bспрос", r"\bdemand", r"восходящ"]
BEARISH = [r"медвеж", r"\bшорт", r"\bshort\b", r"продаж", r"вниз", r"\bbear", r"\bsell", r"паден", r"сопротивлен",
           r"предложен", r"\bsupply", r"нисходящ"]
STOP = {"и", "в", "на", "с", "у", "к", "по", "от", "до", "за", "что", "чтобы", "это", "как", "так", "же", "бы",
        "который", "которая", "которое", "которую", "которые", "если", "то", "а", "но", "или", "мне", "надо",
        "нужно", "должен", "должна", "должно", "был", "была", "было", "быть", "есть", "его", "её", "их", "там",
        "тут", "вот", "уже", "ещё", "еще", "очень", "просто", "только", "паттерн", "паттерна"}
ENDINGS = sorted(["иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ией", "ость", "ости", "ться",
                  "тся", "ение", "ения", "ании", "ание", "ется", "ются", "ала", "ила", "ела", "ыла", "али", "или",
                  "ает", "яет", "ует", "ют", "ут", "ет", "ит", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
                  "ом", "ем", "ам", "ям", "ах", "ях", "ую", "юю", "ов", "ев", "ей", "ешь", "ать", "ять", "ить",
                  "еть", "ал", "ил", "ел", "ыл", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь"], key=len, reverse=True)


def _norm(text):
    text = (text or "").lower().replace("ё", "е").replace("×", "x")
    return re.sub(r"\s+", " ", text)


def stem(word):
    """Основа слова: «снял», «снятие», «снимает» → «сня»/«сним» — сравниваем основы, а не формы."""
    word = re.sub(r"[^\w]", "", word)
    for ending in ENDINGS:
        if len(word) - len(ending) >= 3 and word.endswith(ending):
            return word[:-len(ending)]
    return word


def _tokens(text):
    return [w for w in re.findall(r"[\w]+", _norm(text))]


def _close(a, b):
    """Похожи ли две основы: одна — начало другой, или отличие в 1–2 буквы (опечатка)."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    return min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def phrase_score(clause_stems, phrase):
    """Насколько фраза из банка «есть» в части описания: доля её значимых слов, найденных в тексте."""
    words = [stem(w) for w in _tokens(phrase) if w not in STOP]
    if not words:
        return 0.0
    found = sum(1 for w in words if any(_close(w, s) for s in clause_stems))
    return found / len(words)


def _clauses(text):
    """Части описания: по запятым, точкам, «;», «и», «а также» — каждое условие отдельно."""
    parts = re.split(r"[,.;:!?\n]|\s+и\s+|\s+а\s+также\s+|\s+плюс\s+|\s+\+\s+|\s+а\s+не\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _found(patterns, clause):
    for p in patterns:
        m = re.search(p, clause)
        if m:
            return m
    return None


def _negated(clause, position):
    """Отрицание относится к понятию, только если стоит прямо перед ним
    (между ними — лишь служебные слова: «не должно быть FVG», «без FVG»).
    «не заполнена рядом с ценой» — «не» относится к «заполнена», а не к «рядом с ценой»."""
    before = _tokens(clause[:position])
    for w in reversed(before[-4:]):
        if w in NEGATION_WORDS or w.startswith(("исключ", "никак")):
            return True
        if w not in STOP and w not in FILLER:
            return False
    return False


FILLER = {"сильно", "сильного", "явно", "явного", "особо", "вообще", "совсем", "никакого", "никакой", "должно",
          "должен", "должна", "может", "могут", "бывает", "было", "будет", "быть", "чтоб", "чтобы", "иметь",
          "имеет", "видно", "много", "большого", "большой", "крупного"}


def _match(key, clause, clause_stems):
    """Нашлось ли понятие в части описания: (позиция, уверенность) или None."""
    _, patterns, phrases, _ = CONCEPTS[key]
    m = _found(patterns, clause)
    if m:
        return m.start(), 1.0
    best = max((phrase_score(clause_stems, p) for p in phrases), default=0.0)
    if best >= 0.75:
        first = next((i for i, t in enumerate(_tokens(clause)) if any(_close(stem(t), stem(w))
                      for p in phrases for w in _tokens(p) if w not in STOP)), 0)
        position = len(" ".join(_tokens(clause)[:first]))
        return position, best
    return None


def parse(text, name=""):
    """Описание словами → условия: {"require", "prefer", "forbid", "timeframes", "direction",
    "min_touches", "understood", "unknown", "text"}."""
    raw = text or ""
    text = _norm(raw)
    spec = {"require": [], "prefer": [], "forbid": [], "timeframes": [], "direction": None,
            "min_touches": None, "understood": [], "unknown": [], "text": raw.strip()}
    for tf, patterns in TIMEFRAMES.items():
        if _found(patterns, text):
            spec["timeframes"].append(tf)
    if not spec["timeframes"] and re.search(OLDER, text):
        spec["timeframes"] = ["12h", "1d", "1w"]
    elif not spec["timeframes"] and re.search(YOUNGER, text):
        spec["timeframes"] = ["1h", "4h"]
    up, down = _found(BULLISH, text), _found(BEARISH, text)
    if up and not down:
        spec["direction"] = "up"
    elif down and not up:
        spec["direction"] = "down"
    touches = re.search(r"(\d+|двух|двуx|тр[её]х|четыр[её]х|пяти|шести)\s*\+?\s*(касан|тест|точ|раз)", text)
    if touches:
        word = touches.group(1)
        spec["min_touches"] = int(word) if word.isdigit() else NUMBER_WORDS[word[:3]]
    if re.search(EACH_LINE, text):
        spec["touches_each"] = True
    group = GROUPS.get(name)
    clauses = _clauses(text)
    for clause in clauses:
        hits = _concepts_in(clause, group, name=name)
        known = hits or _found([p for ps in TIMEFRAMES.values() for p in ps] + BULLISH + BEARISH
                               + [OLDER, YOUNGER, r"\bтен", r"фитил"], clause) \
            or re.search(r"касан", clause) or _is_pattern_name(clause, name)
        _add(spec, hits)
        if not known and len(clause) > 3:
            spec["unknown"].append(clause)
    if not (spec["require"] or spec["prefer"] or spec["forbid"]):
        # по частям не нашлось — пробуем всё описание целиком (длинная фраза без запятых)
        _add(spec, _concepts_in(text, group, whole=True, name=name))
        if spec["require"] or spec["prefer"] or spec["forbid"]:
            spec["unknown"] = []
    return spec


def _concepts_in(clause, group, whole=False, name=""):
    """Понятия в части описания: [(понятие, режим)]."""
    stems = [stem(t) for t in _tokens(clause) if t not in STOP]
    hits = []
    for key, (_, _, _, groups) in CONCEPTS.items():
        if group and group not in groups:
            continue                                # к этому паттерну понятие не относится
        found = _match(key, clause, stems)
        if not found:
            continue
        position, _ = found
        negated = _negated(clause, position)
        if key in ("fresh", "untapped") and negated:
            negated = False                         # «не тронут», «не снята» — это и есть само понятие
        if key == "trend" and re.search(r"против\s+тренд", clause):
            negated = not re.search(r"\bне\s+против\s+тренд", clause)     # «не против тренда» = по тренду
            if re.search(r"против\s+тренд\w*\s+(\w+\s+)?(не\s+нуж|не\s+надо|не\s+присыл|нельзя|исключ|не\s+брать|не\s+хочу)", clause) \
                    or re.search(r"\bбез\s+(\w+\s+)?против\s+тренд", clause):
                negated = False                     # «против тренда не нужен» = только по тренду
        if key == "key" and re.search(r"\bне\s+все\b|а\s+не\s+все", clause):
            negated = False                         # «только ключевой, а не все» — это и есть «ключевой»
        mode = "forbid" if negated else ("prefer" if re.search(PREFER, clause) and not re.search(REQUIRE, clause)
                                         else "require")
        hits.append((key, mode))
    keys = {k for k, _ in hits}
    if "inside_ob" in keys:
        hits = [h for h in hits if h[0] != "imbalance"]          # «FVG внутри OB» — одно условие
    if ("untapped", "require") in hits:
        hits = [h for h in hits if h[0] != "swept"]             # «не снята» — одно условие, не два
    if "key" in keys:
        hits = [h for h in hits if h[0] not in ("recent", "trend") or h[1] == "forbid"]   # «последний» = ключевой
    if "partial" in keys:
        hits = [h for h in hits if h[0] != "fresh"]
    if re.search(r"ложн\w*\s+(пробо|выход|вынос)", clause):
        hits = [h for h in hits if h[0] != "confirmed"]          # ложный пробой — это снятие ликвидности
    if name == "FVG":
        hits = [h for h in hits if h[0] != "imbalance" or h[1] == "forbid"]     # «имба» у FVG — это само имя
    return hits


def _add(spec, hits):
    for key, mode in hits:
        if key not in spec["require"] + spec["prefer"] + spec["forbid"]:
            spec[mode].append(key)
            spec["understood"].append((key, mode))


def _is_pattern_name(clause, name):
    words = [r"\bob\b", r"ордер", r"блок", r"\bfvg", r"брейкер", r"breaker", r"митиг", r"\bbos\b", r"\bchoch",
             r"двойн", r"голов", r"плеч", r"треугол", r"клин", r"флаг", r"вымпел", r"канал", r"боков", r"range",
             r"паттерн", r"фигур", r"eqh", r"eql", r"равн", r"зон", r"уровн", r"ищи", r"найди", r"хочу", r"нужн",
             r"определя", r"показыв", r"отмеча", r"рису"]
    return bool(_found(words, clause))


NUMBER_WORDS = {"дву": 2, "тре": 3, "трё": 3, "чет": 4, "пят": 5, "шес": 6}
EACH_LINE = r"сверху\s+и\s+снизу|снизу\s+и\s+сверху|на\s+кажд|кажд\w*\s+(лини|границ|сторон)|обе[иих]?\s+(лини|границ|сторон)|с\s+обеих"


def line_touches(item):
    """Касания каждой линии фигуры: (верхняя, нижняя) или None."""
    points = item.get("touch_points")
    if not points or "upper" not in item or "lower" not in item:
        return None

    def at(line, x):
        x0, y0, x1, y1 = line
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0
    upper = sum(1 for x, y in points if abs(y - at(item["upper"], x)) <= abs(y - at(item["lower"], x)))
    return upper, len(points) - upper


def touches_ok(spec, item):
    """Хватает ли касаний: всего или (если «сверху и снизу») на КАЖДОЙ линии."""
    need = spec.get("min_touches")
    if not need:
        return True
    if spec.get("touches_each"):
        each = line_touches(item)
        if each is not None:
            return min(each) >= need
    touches = item.get("touches")
    if not isinstance(touches, (int, list)):
        return True
    return (touches if isinstance(touches, int) else len(touches)) >= need


def describe(spec):
    """Как бот понял описание — словами, для ответа пользователю."""
    lines = []
    for mode, title in (("require", "обязательно"), ("prefer", "желательно"), ("forbid", "нельзя")):
        keys = [k for k in spec.get(mode) or [] if k in CONCEPTS]
        if keys:
            lines.append(f"{title}: " + "; ".join(CONCEPTS[k][0] for k in keys))
    if spec.get("timeframes"):
        lines.append("таймфреймы: " + ", ".join(tf.upper() for tf in spec["timeframes"]))
    if spec.get("direction"):
        lines.append("направление: " + ("бычий (вверх)" if spec["direction"] == "up" else "медвежий (вниз)"))
    if spec.get("min_touches"):
        lines.append(f"касаний: не меньше {spec['min_touches']}"
                     + (" на каждой линии (сверху и снизу)" if spec.get("touches_each") else ""))
    return lines


def is_empty(spec):
    return not (spec and (spec.get("require") or spec.get("prefer") or spec.get("forbid")
                          or spec.get("timeframes") or spec.get("direction") or spec.get("min_touches")))


def merge(*specs):
    """Несколько описаний одного примера (твоё + выученное по 👍/👎) — в одно."""
    out = {"require": [], "prefer": [], "forbid": [], "timeframes": [], "direction": None,
           "min_touches": None, "understood": [], "unknown": [], "text": ""}
    for spec in specs:
        if not spec:
            continue
        for mode in ("require", "prefer", "forbid"):
            for key in spec.get(mode) or []:
                if key not in out["require"] + out["prefer"] + out["forbid"]:
                    out[mode].append(key)
        out["timeframes"] = out["timeframes"] or list(spec.get("timeframes") or [])
        out["direction"] = out["direction"] or spec.get("direction")
        out["min_touches"] = out["min_touches"] or spec.get("min_touches")
        out["touches_each"] = out.get("touches_each") or spec.get("touches_each")
    return out


# --- проверка находки ---------------------------------------------------------------------------
def _relations(df, item):
    """Связки паттернов, которые нельзя прочитать из одной находки."""
    import quality
    from structure import atr, equal_levels
    out = {}
    if "zone" in item:
        bottom, top = item["zone"]
        size = float(np.nanmean(atr(df).to_numpy())) or 1.0
        levels = [lv for lv in equal_levels(df) if lv.get("swept_at") is None or lv["swept_at"] >= item["start"]]
        out["near_liq"] = any(bottom - 1.5 * size <= lv["level"] <= top + 1.5 * size for lv in levels)
        if str(item.get("kind", "")).endswith("_fvg"):
            from blocks import all_blocks
            out["inside_ob"] = any(b["family"] == "ob" and quality.is_bullish(b) == quality.is_bullish(item)
                                   and b["zone"][0] <= top and bottom <= b["zone"][1]
                                   and abs(b["start"] - item["start"]) <= 6 for b in all_blocks(df))
        else:
            out["inside_ob"] = bool(item.get("imbalance"))
    if item.get("kind") in ("bos", "choch"):
        from structure import structure_events
        events = structure_events(df)
        last = events[-1] if events else None
        out["key"] = bool(last) and last["end"] == item["end"] and last["direction"] == item["direction"]
    return out


def facts(df, item):
    """Что у находки ЕСТЬ: {понятие: True/False/None}. None — к этому паттерну не относится."""
    import quality
    import strength
    strength.evaluate(df, item)
    f = item.get("factors") or {}
    has = lambda k, level=0.5: (f[k] >= level) if k in f else None
    n = len(df)
    end = int(min(item.get("end", item["start"] + item.get("window", 1) - 1), n - 1))
    touches = item.get("touches")
    count = touches if isinstance(touches, int) else (len(touches) if isinstance(touches, list) else None)
    out = {
        "liquidity": item["swept"] if "swept" in item else has("liquidity"),
        "volume": item["loud"] if "loud" in item else has("volume", 0.4),
        "engulf": item.get("engulfed") if "engulfed" in item else has("impulse", 0.6),
        "impulse": has("impulse") if "impulse" in f else has("leg", 0.6) if "leg" in f else has("pole", 0.5),
        "imbalance": (True if str(item.get("kind", "")).endswith("_fvg") else bool(item.get("imbalance")))
        if "zone" in item else None,
        "structure": has("structure", 0.9),
        "discount": has("discount", 0.55),
        "trend": has("trend", 0.9),
        "fresh": has("fresh", 0.9) if "fresh" in f else (item.get("touched_at") is None if "zone" in item else None),
        "body_close": has("close_beyond", 0.3) if "close_beyond" in f else has("body", 0.6),
        "untapped": (item.get("swept_at") is None) if item.get("kind") in ("eqh", "eql") else None,
        "swept": (item.get("swept_at") is not None) if item.get("kind") in ("eqh", "eql") else None,
        "tight": has("tight"),
        "confirmed": bool(item.get("confirmed")) if item.get("points") else (
            item.get("breakout") is not None if ("upper" in item or "box" in item) and "breakout" in item else None),
        "symmetry": has("symmetry", 0.6),
        "major": has("leg", 0.6),
        "internal": (not has("leg", 0.4)) if "leg" in f else None,
        "near_price": (item.get("distance", 1) <= 0.01) if "distance" in item else None,
        "recent": end >= n - 10,
        "many_touches": (count >= 5) if count is not None else None,
        "compression": (not str(item.get("kind", "")).startswith("channel")) if "upper" in item else None,
        "partial": (0.05 <= float(item.get("filled") or 0) < 0.5) if str(item.get("kind", "")).endswith("_fvg") else None,
    }
    if "zone" in item:
        from structure import atr
        bottom, top = item["zone"]
        size = float(np.nanmean(atr(df).to_numpy())) or 1.0
        out["big"] = (top - bottom) / size >= 1.2
        out["small"] = (top - bottom) / size <= 0.6
    elif "upper" in item or "box" in item or item.get("points"):
        out["big"] = has("height", 0.5)
        out["small"] = None if out["big"] is None else not out["big"]
    if item.get("kind") in ("bos", "choch"):
        lows, highs, closes = (df[k].to_numpy(float) for k in ("low", "high", "close"))
        after = slice(end + 1, n)
        if end + 1 < n:
            out["retest"] = bool((lows[after] <= item["level"]).any() if item["direction"] == "up"
                                 else (highs[after] >= item["level"]).any())
        else:
            out["retest"] = False
    if "zone" in item and out.get("fresh") is not None:
        out["retest"] = not out["fresh"]                     # у зоны «ретест» — цена уже возвращалась
    for key, value in _figure_facts(df, item).items():
        if out.get(key) is None:                             # чего не знают общие правила — досчитываем
            out[key] = value
    out.update(_relations(df, item))
    if item.get("kind") in ("eqh", "eql"):                    # у ликвидности — свои значения
        out["liquidity"] = out["swept"]
    out["_bullish"] = quality.is_bullish(item)
    return out


def _line_at(line, x):
    x0, y0, x1, y1 = line
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0


def _figure_breakout(df, item):
    """Пробой фигуры: (номер свечи, "up"/"down", уровень пробоя) или None.
    Треугольник / клин / канал / флаг — закрытие за линией после фигуры; Range — за коробкой;
    дно / ГиП — закрытие за шеей после последней точки."""
    closes = df["close"].to_numpy(float)
    n = len(closes)
    if "upper" in item and "lower" in item:
        x_end = int(max(item["upper"][2], item["lower"][2]))
        for x in range(x_end + 1, n):
            if closes[x] > _line_at(item["upper"], x):
                return x, "up", _line_at(item["upper"], x)
            if closes[x] < _line_at(item["lower"], x):
                return x, "down", _line_at(item["lower"], x)
        return None
    if "box" in item:
        bottom, top = item["box"]
        for x in range(int(item["start"]) + 5, n):
            if item.get("breakout") is None:
                break
            if closes[x] > top and item["breakout"] == "up":
                return x, "up", top
            if closes[x] < bottom and item["breakout"] == "down":
                return x, "down", bottom
        return None
    if item.get("points") and item.get("neckline"):
        import quality
        up = quality.is_bullish(item)
        last = int(max(x for x, _ in item["points"]))
        for x in range(last + 1, n):
            neck = _line_at(item["neckline"], x)
            if (up and closes[x] > neck) or (not up and closes[x] < neck):
                return x, "up" if up else "down", neck
    return None


def _figure_facts(df, item):
    """Условия для фигур, флагов, Range, двойных дн и ГиП — то, что не дают общие правила:
    пробой (подтверждение), закрытие телом, ретест после пробоя, объём, тренд перед флагом,
    ложный вынос и ровность границ у Range, частичная отработка блока."""
    out = {}
    from structure import atr
    n = len(df)
    opens, highs, lows, closes = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    size = float(np.nanmean(atr(df).to_numpy())) or 1e-9
    volume = df["volume"].to_numpy(float) if "volume" in df else None
    start = int(item["start"])
    figure = "upper" in item or "box" in item or bool(item.get("points"))
    if figure:
        hit = _figure_breakout(df, item)
        out["confirmed"] = hit is not None
        out["body_close"] = False
        out["retest"] = False
        if hit is not None:
            x, way, level = hit
            body, candle = abs(closes[x] - opens[x]), highs[x] - lows[x]
            out["body_close"] = bool(candle > 0 and body / candle >= 0.5)
            later = slice(x + 1, n)
            out["retest"] = bool(x + 1 < n and ((lows[later] <= level + 0.3 * size).any() if way == "up"
                                                else (highs[later] >= level - 0.3 * size).any()))
        if volume is not None:
            stop = min(n, start + int(item.get("window", n - start)))
            base = float(np.mean(volume[max(0, start - 20):stop])) or 1e-9
            spot = volume[hit[0]] if hit is not None else float(np.max(volume[max(start, stop - 5):stop]))
            out["volume"] = bool(spot >= 1.3 * base)
    if item.get("pole"):                                   # флаг / вымпел: по тренду, если до флагштока шли туда же
        pole_start = int(item["pole"][0])
        back = closes[max(0, pole_start - 50)]
        up = item.get("direction") == "up"
        out["trend"] = bool((closes[pole_start] > back) == up)
    if "box" in item:                                      # Range
        bottom, top = item["box"]
        height = top - bottom
        span = slice(start, min(n, start + int(item.get("window", n - start))))
        out["liquidity"] = bool(((highs[span] > top + 0.1 * height) & (closes[span] < top)).any()
                                or ((lows[span] < bottom - 0.1 * height) & (closes[span] > bottom)).any())
        points = item.get("touch_points") or []
        mid = (top + bottom) / 2
        upper = [y for _, y in points if y >= mid]
        lower = [y for _, y in points if y < mid]
        if len(upper) >= 2 and len(lower) >= 2:
            out["tight"] = bool(max(np.std(upper), np.std(lower)) <= 0.1 * height)
    if "zone" in item and not str(item.get("kind", "")).endswith("_fvg"):   # блок: цена зашла, но не прошла насквозь
        bottom, top = item["zone"]
        end = int(item.get("end", start))
        after = slice(end + 1, n)
        if end + 1 < n:
            import quality
            if quality.is_bullish(item):
                entered, through = (lows[after] <= top).any(), (closes[after] < bottom).any()
            else:
                entered, through = (highs[after] >= bottom).any(), (closes[after] > top).any()
            out["partial"] = bool(entered and not through)
        else:
            out["partial"] = False
    return out


def check(spec, df, item, interval=None):
    """(подходит?, сколько «желательных» выполнено). Неизвестное для паттерна условие не мешает."""
    if is_empty(spec):
        return True, 0
    if interval and spec.get("timeframes") and interval not in spec["timeframes"]:
        return False, 0
    f = facts(df, item)
    if spec.get("direction"):
        directed = str(item.get("kind", "")).startswith(("bullish", "bearish")) or item.get("direction") \
            or item.get("kind") in ("eqh", "eql") or item.get("points")
        if directed and f["_bullish"] != (spec["direction"] == "up"):
            return False, 0
    for key in spec.get("require", []):
        if f.get(key) is False:
            return False, 0
    for key in spec.get("forbid", []):
        if f.get(key) is True:
            return False, 0
    if not touches_ok(spec, item):
        return False, 0
    return True, sum(1 for key in spec.get("prefer", []) if f.get(key) is True)


def check_any(specs, df, item, interval=None):
    """Несколько примеров с описаниями: находка подходит, если подходит хотя бы под одно."""
    specs = list(specs)
    described = [s for s in specs if not is_empty(s)]
    if not described:
        return True, 0
    results = [check(s, df, item, interval) for s in described]
    ok = [score for good, score in results if good]
    if len(described) < len(specs):                  # есть пример без описания — он ничего не запрещает
        return True, max(ok, default=0)
    return (True, max(ok)) if ok else (False, 0)


def timeframes(specs):
    """Таймфреймы, на которых паттерн вообще искать (None — на всех)."""
    specs = list(specs)
    if not specs or any(is_empty(s) or not s.get("timeframes") for s in specs):
        return None
    return {tf for s in specs for tf in s["timeframes"]}


def present(df, item):
    """Понятия, которые у находки есть / нет — для обучения по 👍/👎: {понятие: 1 / 0}."""
    try:
        f = facts(df, dict(item))
    except Exception:
        return {}
    return {k: int(v) for k, v in f.items() if not k.startswith("_") and v is not None}


# --- понимание: нейросеть + словарь ------------------------------------------------------------------
def understand(text, name="", smart=True):
    """Описание словами → условия. Если на ПК стоит «мозг» (brain.py, своя нейросеть Qwen) —
    понимает любую формулировку; словарь (parse) страхует: что нашёл он, а нейросеть упустила,
    добавляем. Нет нейросети — только словарь."""
    rules = parse(text, name)
    if not smart:                                              # Free или нейросеть занята лимитом — словарь
        return rules
    import brain
    smart = brain.understand(text, name)
    if smart is None:
        return rules
    spec = {**smart, "text": rules["text"], "brain": True}
    taken = set(spec["require"] + spec["prefer"] + spec["forbid"])
    for key, mode in rules["understood"]:
        if key not in taken:                                   # нейросеть пропустила — словарь подскажет
            spec[mode].append(key)
            spec["understood"].append((key, mode))
            taken.add(key)
    spec["timeframes"] = spec["timeframes"] or rules["timeframes"]
    spec["direction"] = spec["direction"] or rules["direction"]
    spec["min_touches"] = spec["min_touches"] or rules["min_touches"]
    spec["touches_each"] = bool(spec.get("touches_each") or rules.get("touches_each"))
    if taken or spec["timeframes"] or spec["direction"]:
        spec["unknown"] = [u for u in spec["unknown"] if u]
    return spec


PATTERN_WORDS = {
    "Order Block": [r"\bob\b", r"\bоб\b", r"ордер\s*блок", r"order\s*block", r"\bордер"],
    "Breaker Block": [r"брейкер", r"breaker"],
    "Mitigation Block": [r"митиг", r"mitigation"],
    "FVG": [r"\bfvg", r"\bфвг", r"имбаланс", r"\bимб", r"imbalance", r"\bгэп"],
    "BOS": [r"\bbos\b", r"\bбос\b", r"слом\w*\s+структ"],
    "CHoCH": [r"\bchoch", r"\bчох", r"смен\w*\s+характер"],
    "EQH / EQL": [r"\beqh\b", r"\beql\b", r"равн\w*\s+(хаи|лои|хай|лой|вершин|дн|максим|миним)"],
    "Range": [r"\brange\b", r"боков", r"проторгов", r"флэт|флет"],
    "Двойное дно": [r"двойн\w*\s+дн"], "Двойная вершина": [r"двойн\w*\s+вершин"],
    "Перевёрнутая ГиП": [r"перев[её]рн\w*\s+(гип|голов)", r"обратн\w*\s+(гип|голов)"],
    "Голова и плечи": [r"голов\w*\s+и\s+плеч", r"\bгип\b"],
    "Треугольник": [r"треугол"], "Клин": [r"\bклин"], "Флаг": [r"\bфлаг"], "Вымпел": [r"вымпел"],
    "Канал": [r"\bканал"],
}


def detect_pattern(text):
    """Какой паттерн упомянут в тексте (по словарю) или None."""
    t = _norm(text)
    for name, patterns in PATTERN_WORDS.items():
        if _found(patterns, t):
            return name
    return None
