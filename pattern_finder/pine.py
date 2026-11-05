"""Разметка для TradingView: скрипт Pine, который рисует найденные паттерны.

Ссылкой рисовать на чужом графике TradingView нельзя, поэтому бот генерирует
маленький индикатор. Пользователь вставляет его в Pine Editor → «Add to chart»,
и линии, зоны и подписи появляются прямо на его графике. Привязка — ко времени
свечей, поэтому разметка встаёт на место на любом таймфрейме этой монеты.

Главное правило — не засорять график. По умолчанию видно только то, что важно СЕЙЧАС:
  • текущие находки (фигуры, зоны, уровни);
  • живые зоны рядом с ценой: по одной ближайшей сверху и снизу на каждом ТФ;
  • панель-сводка в углу: тренд и находки по каждому ТФ.
Прошлые случаи («история») видны сразу, но только полезные:
  • зоны FVG / блоков, которые ещё НЕ отработаны (не закрыты, не пробиты);
  • ликвидность EQH / EQL, которую ещё не сняли;
  • последние 3 слома структуры (BOS / CHoCH) на каждом ТФ;
  • из фигур (каналы, треугольники, клинья, флаги, формы, боковики) — только ПОСЛЕДНЯЯ
    каждого паттерна на каждом ТФ, иначе график превращается в кашу линий.
Выключаются галочкой «Прошлые случаи» в настройках индикатора.
Одна и та же фигура на нескольких ТФ рисуется один раз — со старшего ТФ.
Толщина линий растёт с таймфреймом: 1H — тонкая, 1D — толстая.
"""

from structure import structure_events
from texts import plural, quality_text
from version import VERSION

MAX_OBJECTS = 200          # Pine разрешает до 500 объектов каждого вида, но огромный скрипт
                           # не компилируется («слишком много в одном блоке») — держим запас
MAX_PAST = 50              # прошлых объектов каждого вида — не больше 50 (самые свежие)
APEX_REACH = 2.0           # линии фигуры продлеваем до вершины, но не дальше 2 ширин фигуры
EXTEND_RATIO = 0.1         # если вершины нет (канал) — продлеваем на 10% ширины
NEAREST_ZONES = 2          # живых зон на ТФ: по две ближайшие сверху и снизу от цены
ZONE_RANGE = 0.15          # и не дальше 15% от цены
SAME_PRICE = 0.02          # фигуры разных ТФ совпадают, если концы линий ближе 2%
SAME_CONTEXT = 0.003       # «смена тренда» двух ТФ — одна линия, если цены ближе 0.3%

INTERVAL_SECONDS = {"1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400, "1w": 604800}
WIDTH = {"1h": 1, "4h": 2, "12h": 2, "1d": 3, "1w": 3}
CATEGORY_INPUTS = {
    "figures": ("show_figures", "Фигуры (треугольник, клин, канал, формы)"),
    "zones": ("show_zones", "Зоны (FVG, Order / Breaker / Mitigation)"),
    "levels": ("show_levels", "Уровни (BOS / CHoCH, EQH / EQL)"),
}
CLEAR = "color.new(color.black, 100)"          # прозрачный фон подписи — только текст
BLOCK_SHORT = {"ob": "OB", "breaker": "Breaker", "mitigation": "Mitigation"}


# --- мелочи --------------------------------------------------------------------------------------
def _time(df, i):
    """Время свечи i в миллисекундах. Для i за пределами графика — продлеваем шагом свечи
    (линия треугольника может уходить в будущее, к своей вершине)."""
    times = df["open_time"]
    n = len(df)
    i = max(int(round(i)), 0)
    if i < n:
        return int(times.iloc[i].timestamp() * 1000)
    step = (times.iloc[-1] - times.iloc[-2]).total_seconds() * 1000 if n > 1 else 0
    return int(times.iloc[-1].timestamp() * 1000 + (i - n + 1) * step)


def _price(value):
    """Цена без экспоненты: 0.00001234, а не 1.234e-05."""
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _text(value):
    return str(value).replace('"', "'")


def _outcome_color(item):
    result = item.get("outcome")
    if result is None:
        return "color.gray"
    return "color.green" if result["change"] > 0 else "color.red"


def _slope(line):
    x0, y0, x1, y1 = line
    return (y1 - y0) / max(x1 - x0, 1)


def _line_end(item):
    """До какой свечи вести линии ТЕКУЩЕЙ фигуры: до вершины треугольника/клина
    (если она недалеко впереди), иначе — чуть дальше последней свечи."""
    _, _, x1, u1 = item["upper"]
    _, _, _, l1 = item["lower"]
    width = max(x1 - min(item["upper"][0], item["lower"][0]), 1)
    up_slope, lo_slope = _slope(item["upper"]), _slope(item["lower"])
    if up_slope < lo_slope:                       # линии сходятся
        apex = x1 + (u1 - l1) / (lo_slope - up_slope)
        if x1 <= apex <= x1 + APEX_REACH * width:
            return apex
    return x1 + EXTEND_RATIO * width


def _at(line, x):
    return line[1] + _slope(line) * (x - line[0])


def category(item):
    if "upper" in item:
        return "figures"
    if "zone" in item:
        return "zones"
    if "level" in item:
        return "levels"
    return "figures"                               # паттерн по форме (двойное дно и т. п.)


def is_alive(item):
    """Зона или уровень ещё работают: FVG не закрыт, блок не пробит, ликвидность не снята."""
    if "zone" in item:
        if item.get("family") in BLOCK_SHORT:            # блок отработан после первого возвращения цены
            return item.get("broken_at") is None and item.get("touched_at") is None
        return item.get("filled_at") is None
    if item.get("kind") in ("eqh", "eql"):
        return item.get("swept_at") is None
    return False


def _bullish(item):
    return str(item.get("kind", "")).startswith("bullish") or item.get("direction") == "up"


def _label(df, x, y, text, color, style="label.style_label_left", size="size.small"):
    return ("label", f'label.new({_time(df, x)}, {_price(y)}, "{_text(text)}", xloc=xloc.bar_time, '
                     f"style={style}, color={CLEAR}, textcolor={color}, size={size})")


def _line(df, x0, y0, x1, y1, color, width, style="line.style_solid", extend="extend.none"):
    return ("line", f"line.new({_time(df, x0)}, {_price(y0)}, {_time(df, x1)}, {_price(y1)}, "
                    f"xloc=xloc.bar_time, extend={extend}, color={color}, width={width}, style={style})")


def _box(df, x0, top, x1, bottom, color, alpha, text="", extend="extend.none", border="line.style_solid"):
    return ("box", f"box.new({_time(df, x0)}, {_price(top)}, {_time(df, x1)}, {_price(bottom)}, "
                   f"xloc=xloc.bar_time, extend={extend}, border_color=color.new({color}, 70), "
                   f"border_style={border}, bgcolor=color.new({color}, {alpha}), "
                   f'text="{_text(text)}", text_size=size.tiny, text_color={color}, '
                   f"text_halign=text.align_right, text_valign=text.align_top)")


# --- рисование одной находки ---------------------------------------------------------------------
def draw_item(df, item, mode, tag="", width=1):
    """Pine-строки для одной находки: [(вид, код)], вид — line / box / label.
    mode: "now" — найдено сейчас, "alive" — живая зона рядом с ценой, "past" — история."""
    out = []
    name = f"{tag}{item['name']}"
    now = mode == "now"
    color = "color.green" if _bullish(item) else "color.red"

    if item.get("kind") == "range":                              # боковик: коробка + POC
        bottom, top = item["box"]
        last = min(item.get("end", item["start"] + item["window"] - 1), len(df) - 1)
        alive = now and not item.get("breakout")
        right = len(df) - 1 if alive else last
        out.append(_box(df, item["start"], top, right, bottom, "color.gray", 88 if mode != "past" else 94,
                        text=f"{tag}Range" if mode != "past" else "",
                        extend="extend.right" if alive else "extend.none"))
        if item.get("poc") is not None and mode != "past":
            out.append(_line(df, item["start"], item["poc"], right, item["poc"], "color.orange", width,
                             style="line.style_dashed"))
            out.append(_label(df, right, item["poc"], "POC", "color.orange", size="size.tiny"))
        return out

    if "upper" in item:                                          # треугольник, клин, канал, флаг
        if item.get("pole"):                                     # флагшток
            x0, y0, x1, y1 = item["pole"]
            out.append(_line(df, x0, y0, x1, y1, "color.purple" if now else "color.new(color.purple, 60)",
                             width + 1 if now else 1))
        end = _line_end(item) if now else item["upper"][2]
        for key, side in (("upper", "color.red"), ("lower", "color.green")):
            line = item[key]
            line_color = side if now else f"color.new({side}, 60)"
            out.append(_line(df, line[0], line[1], end, _at(line, end), line_color, width + 1 if now else 1))
        if now:                                                  # подпись — только ТФ и название
            out.append(_label(df, end, _at(item["upper"], end), short_name(item, tag), "color.blue"))
            for x, y in item.get("touch_points", []):              # точки касаний — маленькие
                out.append(_label(df, x, y, "●", "color.blue", style="label.style_label_center",
                                  size="size.tiny"))
        else:
            out.append(_label(df, item["upper"][2], item["upper"][3], short_name(item, tag),
                              "color.gray", size="size.tiny"))

    elif item.get("kind") in ("bos", "choch"):                   # слом структуры
        text = f"{tag}{'BOS' if item['kind'] == 'bos' else 'CHoCH'}"
        line_color = color if now else f"color.new({color}, 25)"        # прошлые сломы — тоже хорошо видны
        out.append(_line(df, item["start"], item["level"], item["end"], item["level"], line_color,
                         max(width, 2), style="line.style_dashed"))
        style = "label.style_label_down" if item["direction"] == "up" else "label.style_label_up"
        out.append(_label(df, (item["start"] + item["end"]) / 2, item["level"], text, line_color,
                          style=style, size="size.small"))

    elif item.get("kind") == "context":                          # уровень смены тренда
        up = item["direction"] == "up"
        line_color = "color.new(color.red, 10)" if up else "color.new(color.green, 10)"
        out.append(_line(df, item["start"], item["level"], len(df) - 1, item["level"], line_color, 2,
                         style="line.style_dashed", extend="extend.right"))
        if tag and item.get("also"):
            tag = " · ".join(t.upper() for t in item["also"]) + " · " + tag
        text = f"{tag}смена тренда {'↓ ниже' if up else '↑ выше'}"
        out.append(_label(df, len(df) - 1, item["level"], text, line_color,
                          style="label.style_label_up" if up else "label.style_label_down", size="size.small"))

    elif item.get("kind") in ("eqh", "eql"):                     # ликвидность
        alive = item.get("swept_at") is None
        right = len(df) - 1 if alive else item["swept_at"]
        line_color = "color.orange" if mode != "past" else "color.new(color.orange, 60)"
        out.append(_line(df, item["start"], item["level"], right, item["level"], line_color, width,
                         style="line.style_dotted", extend="extend.right" if alive else "extend.none"))
        if mode != "past":
            text = f"{tag}{'EQH' if item['kind'] == 'eqh' else 'EQL'}"
            style = "label.style_label_down" if item["kind"] == "eqh" else "label.style_label_up"
            out.append(_label(df, item["end"], item["level"], text, "color.orange", style=style))

    elif "zone" in item:                                         # FVG и блоки
        bottom, top = item["zone"]
        block = item.get("family") in BLOCK_SHORT
        short = BLOCK_SHORT[item["family"]] if block else "FVG"
        border = "line.style_dashed" if block and item["family"] != "ob" else "line.style_solid"
        if mode == "past":
            closed = item.get("broken_at") if block else item.get("filled_at")
            right = closed if closed is not None else len(df) - 1
            out.append(_box(df, item["start"], top, right, bottom, color, 94, border=border))
        else:
            alpha = 85 if now else 90
            if item.get("grade") == "strong":
                alpha -= 10                                   # сильная зона — заливка плотнее (без цифр)
            out.append(_box(df, item["start"], top, len(df) - 1, bottom, color, alpha,
                            text=f"{tag}{short}", extend="extend.right", border=border))

    elif item.get("points"):                                     # двойное дно / ГиП — ломаная по точкам + шея
        line_color = "color.blue" if now else "color.new(color.gray, 40)"
        points = item["points"]
        for (xa, ya), (xb, yb) in zip(points, points[1:]):
            out.append(_line(df, xa, ya, xb, yb, line_color, width + 1 if now else 1))
        x0, y0, x1, y1 = item["neckline"]
        out.append(_line(df, x0, y0, x1, y1, "color.orange" if now else "color.new(color.orange, 60)", width,
                         style="line.style_dashed"))
        x_top, y_top = max(points, key=lambda p: p[1])
        out.append(_label(df, x_top, y_top, short_name(item, tag), line_color,
                          style="label.style_label_down", size="size.small" if now else "size.tiny"))
        return out

    else:                                                        # паттерн по форме без точек — рамка
        x0 = item["start"]
        x1 = min(x0 + item["window"] - 1, len(df) - 1)
        segment = df.iloc[x0:x1 + 1]
        top, bottom = segment["high"].max(), segment["low"].min()
        box_color = "color.blue" if now else "color.gray"
        out.append(_box(df, x0, top, x1, bottom, box_color, 90 if now else 95))
        out.append(_label(df, x0, top, short_name(item, tag), box_color,
                          style="label.style_label_lower_left", size="size.small" if now else "size.tiny"))
    return out


# --- что показывать --------------------------------------------------------------------------------
def _normalize(groups, interval):
    """Группы бывают (df, находки, история) — тогда таймфрейм = interval —
    или (df, находки, история, таймфрейм)."""
    return [g if len(g) == 4 else (*g, interval) for g in groups]


def _end_ms(df, item):
    return _time(df, min(item["start"] + item["window"] - 1, len(df) - 1))


def _zone_mid(item):
    if "zone" in item:
        return sum(item["zone"]) / 2
    return item["level"]


def _nearest_alive(df, items):
    """Из живых зон и уровней ликвидности оставляет ближайшие к цене:
    по NEAREST_ZONES сверху и снизу для каждой категории, не дальше ZONE_RANGE."""
    price = float(df["close"].iloc[-1])
    chosen = []
    for cat in ("zones", "levels"):
        alive = [it for it in items if category(it) == cat and is_alive(it)
                 and abs(_zone_mid(it) - price) / price <= ZONE_RANGE]
        above = sorted((it for it in alive if _zone_mid(it) >= price), key=_zone_mid)
        below = sorted((it for it in alive if _zone_mid(it) < price), key=_zone_mid, reverse=True)
        chosen += above[:NEAREST_ZONES] + below[:NEAREST_ZONES]
    return chosen


def _same_figure(a, b):
    """Одна и та же фигура на двух ТФ: тот же отрезок времени и концы линий почти на тех же ценах."""
    (dfa, ia), (dfb, ib) = a, b
    if "upper" not in ia or "upper" not in ib:
        return False
    a0, a1 = _time(dfa, ia["start"]), _end_ms(dfa, ia)
    b0, b1 = _time(dfb, ib["start"]), _end_ms(dfb, ib)
    overlap = max(0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1)
    close = all(abs(ia[k][3] - ib[k][3]) / abs(ib[k][3]) <= SAME_PRICE for k in ("upper", "lower"))
    return overlap >= 0.5 and close


SAME_ZONE = 0.8             # зоны двух ТФ — «одна и та же», если совпадают на 80% (от большей)


def _same_zone(a, b):
    """Одна и та же зона на двух ТФ: то же направление и почти те же цены.
    Маленькая зона младшего ТФ ВНУТРИ большой зоны старшего — это НЕ дубль: она точнее
    показывает, где реакция (4H FVG внутри 12H FVG рисуем оба)."""
    (_, ia), (_, ib) = a, b
    if "zone" not in ia or "zone" not in ib or _bullish(ia) != _bullish(ib):
        return False
    low, high = max(ia["zone"][0], ib["zone"][0]), min(ia["zone"][1], ib["zone"][1])
    larger = max(ia["zone"][1] - ia["zone"][0], ib["zone"][1] - ib["zone"][0])
    return larger > 0 and (high - low) / larger >= SAME_ZONE


PAST_STRUCTURE = 1          # на ТФ — только последний слом (он задаёт текущий тренд)
PAST_FIGURES = 0            # прошлые (уже отработанные) фигуры и боковики не рисуем — пользы нет


RECENT_FIGURE_BARS = 60    # прошлая фигура интересна, только если закончилась недавно


def worth_showing(items, n=None):
    """Из прошлых случаев (кроме зон и ликвидности — их выбирает _nearest_alive) оставляет:
    последние 3 слома структуры и одну последнюю фигуру каждого паттерна, если она
    закончилась недавно (n — длина графика; без n недавность не проверяем)."""
    end = lambda it: it["start"] + it.get("window", 1)
    kept = []
    by_name = {}
    for it in items:
        if it.get("kind") in ("bos", "choch"):
            by_name.setdefault(("structure", it["name"]), []).append(it)
        elif category(it) == "figures" and (n is None or end(it) >= n - RECENT_FIGURE_BARS):
            by_name.setdefault(("figure", it["name"]), []).append(it)
    structure = sorted((it for (group, _), found in by_name.items() if group == "structure" for it in found),
                       key=lambda it: it.get("end", it["start"] + it.get("window", 1) - 1))
    kept += structure[-PAST_STRUCTURE:] if PAST_STRUCTURE else []      # BOS и CHoCH вместе — один последний
    for (group, _), found in by_name.items():
        if group == "figure" and PAST_FIGURES:
            kept += sorted(found, key=end)[-PAST_FIGURES:]
    return kept


MAX_PAST_FIGURES = 1        # прошлых фигур на графике (все ТФ вместе)


def _limit_figures(layers):
    """Прошлые фигуры всех ТФ: не больше двух — сначала каналы, со старшего ТФ; не на том же
    участке времени, что текущие фигуры или уже выбранные (иначе каша из линий)."""
    from scanner import FIGURE_PRIORITY, figure_family, is_figure
    spans = [(_time(df, it["start"]), _end_ms(df, it)) for tf in layers for df, it in layers[tf]["now"]
             if is_figure(it)]
    past = [(tf, df, it) for tf in layers for df, it in layers[tf]["past"] if is_figure(it)]
    past.sort(key=lambda e: (FIGURE_PRIORITY[figure_family(e[2])], INTERVAL_SECONDS[e[0]], _end_ms(e[1], e[2])),
              reverse=True)
    keep = set()
    for tf, df, it in past:
        start, end = _time(df, it["start"]), _end_ms(df, it)
        if len(keep) < MAX_PAST_FIGURES and not any(start <= b and end >= a for a, b in spans):
            keep.add(id(it))
            spans.append((start, end))
    for tf in layers:
        layers[tf]["past"] = [(df, it) for df, it in layers[tf]["past"] if not is_figure(it) or id(it) in keep]


def plan(groups):
    """Раскладывает находки по слоям: {таймфрейм: {"now": [...], "alive": [...], "past": [...]}}.
    Каждая запись — (df, находка)."""
    tfs = sorted({g[3] for g in groups}, key=lambda i: INTERVAL_SECONDS[i])
    layers = {tf: {"now": [], "alive": [], "past": []} for tf in tfs}
    for df, items, history, tf in groups:
        if not history:
            layers[tf]["now"] += [(df, it) for it in items]
    for df, items, history, tf in groups:
        if history:
            # зоны и ликвидность: только НЕ задетые и рядом с ценой (по 2 сверху и снизу) —
            # от них может быть реакция; старые далёкие зоны — шум
            current_zones = {it.get("zone") for _, it in layers[tf]["now"] if "zone" in it}
            alive = [it for it in _nearest_alive(df, items) if it.get("zone") not in current_zones]
            layers[tf]["alive"] += [(df, it) for it in alive]
            shown = worth_showing(items, len(df))
            layers[tf]["past"] += [(df, it) for it in shown]
            if any(it.get("kind") in ("bos", "choch") for it in items):
                from structure import context_level
                level = context_level(df)                  # где сменится тренд — главный уровень ТФ
                if level is not None:
                    layers[tf]["past"].append((df, {**level, "name": "Смена тренда"}))

    _limit_figures(layers)
    # одна фигура / зона на нескольких ТФ — оставляем её на старшем ТФ.
    # Текущие находки («Сейчас найдено» в алерте) не убираем никогда: что написано — то и нарисовано
    for i, low_tf in enumerate(tfs):
        for high_tf in tfs[i + 1:]:
            higher = layers[high_tf]["now"] + layers[high_tf]["alive"]
            layers[low_tf]["alive"] = [
                entry for entry in layers[low_tf]["alive"]
                if not any(_same_figure(entry, h) or _same_zone(entry, h) for h in higher)
            ]
    _merge_context(layers, tfs)
    return layers


def _merge_context(layers, tfs):
    """«Смена тренда» на одной цене у нескольких ТФ — одна линия с подписью «4H · 12H»,
    а не надписи друг на друге."""
    for i, low_tf in enumerate(tfs):
        for high_tf in tfs[i + 1:]:
            highs = [it for _, it in layers[high_tf]["past"] if it.get("kind") == "context"]
            kept = []
            for df, it in layers[low_tf]["past"]:
                twin = next((h for h in highs if it.get("kind") == "context" and h["direction"] == it["direction"]
                             and abs(h["level"] - it["level"]) <= SAME_CONTEXT * abs(h["level"])), None)
                if twin is None:
                    kept.append((df, it))
                else:
                    twin["also"] = sorted(set(twin.get("also", [])) | {low_tf} | set(it.get("also", [])),
                                          key=lambda t: INTERVAL_SECONDS[t])
            layers[low_tf]["past"] = kept


# --- панель-сводка ----------------------------------------------------------------------------------
def _trend(groups, tf):
    """Тренд ТФ по последнему слому структуры: ↑, ↓ или —."""
    for df, _, history, g_tf in groups:
        if g_tf == tf and history:
            events = structure_events(df)
            if events:
                return "↑ вверх" if events[-1]["direction"] == "up" else "↓ вниз"
    return "—"


SHORT_NAMES = {"Order Block": "OB", "Breaker Block": "Breaker", "Mitigation Block": "Mitigation",
               "EQH / EQL": "EQH/EQL"}


def short_name(item, tag=""):
    """Подпись на графике: только ТФ и элемент — «1D OB ↑», «4H Канал» (без цифр и процентов)."""
    name = SHORT_NAMES.get(item["name"], item["name"])
    kind = str(item.get("kind", ""))
    directed = kind.startswith(("bullish", "bearish")) or kind in ("bos", "choch")
    arrow = (" ↑" if _bullish(item) else " ↓") if directed else ""
    return f"{tag}{name}{arrow}"


def _short(item):
    return short_name(item)


def now_items(groups, tf):
    """Все находки «сейчас» на ТФ — как в алерте (до того, как одинаковые зоны разных ТФ
    схлопнулись на графике в одну)."""
    return [it for _, items, history, g_tf in groups if g_tf == tf and not history for it in items]


def _table(groups, layers, tfs):
    rows = []
    for tf in tfs:
        found = ", ".join(dict.fromkeys(_short(it) for it in now_items(groups, tf))) or "—"
        rows.append((tf.upper(), _trend(groups, tf), found))
    code = [
        f"pf_table = table.new(position.top_right, 3, {len(rows) + 1}, "
        "bgcolor=color.new(color.black, 25), border_width=1, border_color=color.new(color.gray, 70))",
    ]
    for col, text in enumerate(("ТФ", "Тренд", "Сейчас найдено")):
        code.append(f'table.cell(pf_table, {col}, 0, "{text}", text_color=color.gray, '
                    "text_size=size.small, text_halign=text.align_left)")
    for row, values in enumerate(rows, start=1):
        for col, text in enumerate(values):
            color = "color.white"
            if col == 1 and text.startswith("↑"):
                color = "color.green"
            elif col == 1 and text.startswith("↓"):
                color = "color.red"
            code.append(f'table.cell(pf_table, {col}, {row}, "{_text(text)}", text_color={color}, '
                        "text_size=size.small, text_halign=text.align_left)")
    return code


# --- скрипт целиком -------------------------------------------------------------------------------
def pine_script(symbol, interval, groups, history_default=True):
    """Код индикатора Pine v5 для одной монеты и всех её таймфреймов.

    groups — список (df, находки, это_история[, таймфрейм]).
    Если объектов больше лимита Pine, оставляем все текущие, потом живые зоны,
    потом самые свежие прошлые."""
    groups = _normalize(groups, interval)
    layers = plan(groups)
    tfs = list(layers)
    multi = len(tfs) > 1

    objects = []                      # (таймфрейм, слой, категория, приоритет, время, вид, код)
    priority = {"now": 2, "alive": 1, "past": 0}
    for tf in tfs:
        tag = f"{tf.upper()} " if multi else ""
        for mode in ("now", "alive", "past"):
            for df, item in layers[tf][mode]:
                when = _end_ms(df, item)
                for kind, code in draw_item(df, item, mode, tag, WIDTH[tf]):
                    objects.append((tf, mode, category(item), priority[mode], when, kind, code))
    kept = []
    for kind in ("line", "box", "label"):
        same = sorted((o for o in objects if o[5] == kind), key=lambda o: (o[3], o[4]), reverse=True)
        current = [o for o in same if o[1] != "past"]
        past = [o for o in same if o[1] == "past"][:MAX_PAST]
        kept += (current + past)[:MAX_OBJECTS]
    kept_ids = set(map(id, kept))

    title = " ".join(tf.upper() for tf in tfs) or interval.upper()
    head = [
        "//@version=5",
        f'indicator("Pattern Finder v{VERSION} · {symbol} {title}", overlay=true, '
        "max_lines_count=500, max_boxes_count=500, max_labels_count=500)",
        'show_table = input.bool(true, "Панель-сводка", group="Что показывать")',
    ]
    head += [f'{var} = input.bool(true, "{text}", group="Что показывать")'
             for var, text in CATEGORY_INPUTS.values()]
    history_flag = "true" if history_default else "false"
    head.append(f'show_history = input.bool({history_flag}, "Прошлые случаи (история)", group="Что показывать")')
    if multi:
        head.append('only_higher = input.bool(true, "Только ТФ не ниже графика", group="Таймфреймы")')
        head += [f'show_{tf} = input.bool(true, "{tf.upper()}", group="Таймфреймы")' for tf in tfs]
        head.append("chart_seconds = timeframe.in_seconds()")

    # каждый таймфрейм — СВОЙ блок верхнего уровня: в одном огромном блоке
    # TradingView не компилирует скрипт (лимит на размер одного блока)
    body = []
    drawn = 0
    for tf in tfs:
        block = []
        for mode in ("now", "alive", "past"):
            for cat, (var, _) in CATEGORY_INPUTS.items():
                lines = [o[6] for o in objects
                         if o[0] == tf and o[1] == mode and o[2] == cat and id(o) in kept_ids]
                if not lines:
                    continue
                block.append(f"if {var} and show_history" if mode == "past" else f"if {var}")
                block += ["    " + line for line in lines]
                drawn += len(lines)
        if not block:
            continue
        if multi:
            body.append(f"if barstate.islast and show_{tf} and "
                        f"(not only_higher or chart_seconds <= {INTERVAL_SECONDS[tf]})")
        else:
            body.append("if barstate.islast")
        body += ["    " + line for line in block]
    if not drawn:
        body += ["if barstate.islast", '    label.new(bar_index, high, "Паттерны не найдены")']
    body.append("if barstate.islast and show_table")
    body += ["    " + line for line in _table(groups, layers, tfs)]
    return "\n".join(head + body) + "\n"


def pine_help(symbol, interval, groups, selected=False, history_default=True, howto=True):
    """Подпись к файлу разметки: что найдено СЕЙЧАС (по таймфреймам) и что скрыто.
    selected — разметка только выбранных паттернов (кнопка «🧩 Паттерны»)."""
    groups = _normalize(groups, interval)
    layers = plan(groups)
    tfs = list(layers)
    now = {tf: [it["name"] for _, it in layers[tf]["now"]] for tf in tfs}
    if len(tfs) > 1:
        found = "; ".join(f"{tf.upper()}: " + ", ".join(dict.fromkeys(names))
                          for tf, names in now.items() if names)
    else:
        found = ", ".join(dict.fromkeys(n for names in now.values() for n in names))
    past = sum(len(layers[tf]["past"]) for tf in tfs)
    title = f"Разметка {symbol}"
    if len(tfs) > 1:
        title += f" — таймфреймы {', '.join(t.upper() for t in tfs)} в одном файле"
    cases = plural(past, 'прошлый случай', 'прошлых случая', 'прошлых случаев')
    if selected:
        head = f"{title}\nТолько выбранное: {found or '—'}.\n"
        if history_default:
            head += f"Прошлые случаи этих паттернов включены: {cases}.\n"
    else:
        head = (f"{title}\nСейчас найдено: {found or '—'}.\n"
                "На графике: текущие находки, неотработанные зоны и ликвидность, последние сломы структуры "
                f"и последние фигуры ({cases}), панель-сводка в правом верхнем углу.\n")
    if not history_default:
        head += (f"Скрыто: {cases} — включить: шестерёнка индикатора → Inputs → «Прошлые случаи».\n")
    if len(tfs) > 1:
        head += "Видны ТФ не ниже открытого графика; чем толще линия — тем старше ТФ.\n"
    if not howto:                                  # коротко — для подписи к файлу
        return head
    return head + "\n" + PINE_HELP.format(symbol=symbol)


PINE_HELP = (
    "Как читать: красная линия — сопротивление, зелёная — поддержка, синие точки — касания;\n"
    "зоны: зелёные — бычьи, красные — медвежьи; пунктирная рамка — Breaker/Mitigation;\n"
    "оранжевые точки EQH/EQL — ликвидность ($$$ — ещё не снята).\n\n"
    "Как включить в TradingView:\n"
    "1. Открой график {symbol} (биржа Binance)\n"
    "2. Удали старые индикаторы Pattern Finder (корзина рядом с названием)\n"
    "3. Внизу открой Pine Editor, удали всё и вставь код из файла\n"
    "4. Нажми «Add to chart»"
)
