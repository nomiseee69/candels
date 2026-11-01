"""Разметка для TradingView: скрипт Pine, который рисует найденные паттерны.

Ссылкой рисовать на чужом графике TradingView нельзя, поэтому бот генерирует
маленький индикатор. Пользователь вставляет его в Pine Editor → «Add to chart»,
и линии, зоны и подписи появляются прямо на его графике. Привязка — ко времени
свечей, поэтому разметка встаёт на место на любом таймфрейме этой монеты.

Рисуется всё сразу: прошлые случаи (цвет = что было после) и текущие находки (синие).

Один скрипт на все таймфреймы монеты: у каждого ТФ свой переключатель в настройках
индикатора, а по умолчанию на графике видны только ТФ не ниже открытого
(на 1H видно 1H, 4H и 1D; на 1D — только 1D, иначе мелочь закроет всё).
"""

from texts import plural, quality_text
from version import VERSION

# Pine разрешает не больше 500 объектов каждого вида на графике
MAX_OBJECTS = 450
APEX_REACH = 2.0       # линии фигуры продлеваем до вершины, но не дальше 2 ширин фигуры
EXTEND_RATIO = 0.1     # если вершины нет (канал) — продлеваем на 10% ширины


def _time(df, i):
    """Время свечи i в миллисекундах. Для i за пределами графика — продлеваем шагом свечи
    (линия треугольника может уходить в будущее, к своей вершине)."""
    times = df["open_time"]
    n = len(df)
    i = int(round(i))
    if i < 0:
        i = 0
    if i < n:
        return int(times.iloc[i].timestamp() * 1000)
    step = (times.iloc[-1] - times.iloc[-2]).total_seconds() * 1000 if n > 1 else 0
    return int(times.iloc[-1].timestamp() * 1000 + (i - n + 1) * step)


def _price(value):
    """Цена без экспоненты: 0.00001234, а не 1.234e-05."""
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _outcome_color(item):
    result = item.get("outcome")
    if result is None:
        return "color.orange"
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


def _label_text(df, item, name, history):
    """«СЕЙЧАС · Треугольник · 6 касаний» или «был · Канал · 21.10» — чтобы на графике
    сразу было видно, что найдено сейчас, а что — прошлые случаи."""
    if history:
        end = min(item["start"] + item["window"] - 1, len(df) - 1)
        return f"был · {name} · {df['open_time'].iloc[end]:%d.%m.%y}"
    quality = quality_text(item)
    return f"СЕЙЧАС · {name}" + (f" · {quality}" if quality else "")


def _draw_item(df, item, history, tag=""):
    """Строки Pine для одной находки. Возвращает (линии, рамки, подписи).
    tag — таймфрейм («4H»), если в скрипте их несколько: он пишется в подписях."""
    lines, boxes, labels = [], [], []
    name = item["name"].replace('"', "'")
    if tag:
        name = f"{tag} {name}"
    prefix = f"{tag} " if tag else ""
    width = 1 if history else 2

    if item.get("kind") in ("bos", "choch"):                   # слом структуры: уровень → свеча слома
        color = "color.green" if item["direction"] == "up" else "color.red"
        level = _price(item["level"])
        lines.append(
            f"line.new({_time(df, item['start'])}, {level}, {_time(df, item['end'])}, {level}, "
            f"xloc=xloc.bar_time, color={color}, style=line.style_dashed, width={width})"
        )
        middle = (item["start"] + item["end"]) / 2
        style = "label.style_label_down" if item["direction"] == "up" else "label.style_label_up"
        text = prefix + ("BOS" if item["kind"] == "bos" else "CHoCH")
        labels.append((middle, item["level"], text, color, style, "color.new(color.black, 100)"))

    elif item.get("kind") in ("eqh", "eql"):                    # равные хаи/лои: уровень ликвидности
        color = "color.orange"
        level = _price(item["level"])
        swept = item.get("swept_at")
        right, extend = (swept, "extend.none") if swept is not None else (len(df) - 1, "extend.right")
        lines.append(
            f"line.new({_time(df, item['start'])}, {level}, {_time(df, right)}, {level}, "
            f"xloc=xloc.bar_time, extend={extend}, color={color}, style=line.style_dotted, width={width})"
        )
        up = item["kind"] == "eqh"
        text = prefix + ("EQH" if up else "EQL") + ("" if swept is not None else " $$$")
        style = "label.style_label_down" if up else "label.style_label_up"
        labels.append((item["end"], item["level"], text, color, style, "color.new(color.black, 100)"))

    elif item.get("family") in ("ob", "breaker", "mitigation"):  # OB / Breaker / Mitigation
        bottom, top = item["zone"]
        color = "color.green" if item["kind"].startswith("bullish") else "color.red"
        broken = item.get("broken_at")
        right, extend, alpha = (broken, "extend.none", 92) if broken is not None else (len(df) - 1, "extend.right", 82)
        text = prefix + {"ob": "OB", "breaker": "Breaker", "mitigation": "Mitigation"}[item["family"]]
        style = "line.style_solid" if item["family"] == "ob" else "line.style_dashed"
        boxes.append(
            f"box.new({_time(df, item['start'])}, {_price(top)}, {_time(df, right)}, {_price(bottom)}, "
            f"xloc=xloc.bar_time, extend={extend}, border_color={color}, border_style={style}, "
            f'bgcolor=color.new({color}, {alpha}), text="{text}", text_size=size.tiny, '
            f"text_color={color}, text_halign=text.align_left)"
        )

    elif "upper" in item:                                         # треугольник, клин, канал
        # прошлая фигура заканчивается там, где закончилась; текущая — тянется к вершине
        end = item["upper"][2] if history else _line_end(item)
        color = _outcome_color(item) if history else "color.blue"
        # сплошные линии: прошлые — тоньше и чуть прозрачнее, текущая — яркая и толстая
        style = "color.new({}, 35), width=1" if history else "{}, width=2"
        for key, side in (("upper", "color.red"), ("lower", "color.green")):
            x0, y0 = item[key][0], item[key][1]
            y_end = y0 + _slope(item[key]) * (end - x0)
            lines.append(
                f"line.new({_time(df, x0)}, {_price(y0)}, {_time(df, end)}, {_price(y_end)}, "
                f"xloc=xloc.bar_time, color={style.format(side)})"
            )
        x0, top = item["upper"][0], item["upper"][1]
        labels.append((x0, top, _label_text(df, item, name, history), color))
        if not history:                                         # точки касаний — почему это фигура
            for x, y in item.get("touch_points", []):
                labels.append((x, y, "", "color.blue", "label.style_circle", "color.new(color.blue, 0)"))

    elif "zone" in item:                                        # FVG
        bottom, top = item["zone"]
        color = "color.green" if item.get("kind") == "bullish_fvg" else "color.red"
        filled_at = item.get("filled_at")
        if filled_at is None:                                   # открыт — тянем вправо
            right, extend, alpha, text = len(df) - 1, "extend.right", 80, prefix + "FVG"
        else:                                                   # закрыт — до свечи закрытия
            right, extend, alpha, text = filled_at, "extend.none", 92, ""
        boxes.append(
            f"box.new({_time(df, item['start'])}, {_price(top)}, {_time(df, right)}, {_price(bottom)}, "
            f"xloc=xloc.bar_time, extend={extend}, border_color=color.new({color}, 60), "
            f'bgcolor=color.new({color}, {alpha}), text="{text}", text_size=size.tiny, '
            f"text_color={color})"
        )

    else:                                                       # паттерн по форме — рамка
        x0 = item["start"]
        x1 = min(x0 + item["window"] - 1, len(df) - 1)
        segment = df.iloc[x0:x1 + 1]
        top, bottom = segment["high"].max(), segment["low"].min()
        color = _outcome_color(item) if history else "color.blue"
        boxes.append(
            f"box.new({_time(df, x0)}, {_price(top)}, {_time(df, x1)}, {_price(bottom)}, "
            f"xloc=xloc.bar_time, border_color={color}, bgcolor=color.new({color}, 85))"
        )
        labels.append((x0, top, _label_text(df, item, name, history), color))

    label_lines = []
    for x, y, text, color, *custom in labels:
        # подпись-плашка (фигуры) или просто текст над/под уровнем (BOS, EQH)
        style, background = custom if custom else ("label.style_label_down", color)
        text_color = "color.white" if not custom else color
        label_lines.append(
            f'label.new({_time(df, x)}, {_price(y)}, "{text}", xloc=xloc.bar_time, '
            f"style={style}, color={background}, textcolor={text_color}, "
            f"size={'size.small' if text else 'size.tiny'})"
        )
    return lines, boxes, label_lines


INTERVAL_SECONDS = {"1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400, "1w": 604800}


def _normalize(groups, interval):
    """Группы бывают (df, находки, история) — тогда таймфрейм = interval —
    или (df, находки, история, таймфрейм)."""
    return [g if len(g) == 4 else (*g, interval) for g in groups]


def _end_time(df, item):
    return _time(df, min(item["start"] + item["window"] - 1, len(df) - 1))


def pine_script(symbol, interval, groups):
    """Код индикатора Pine v5 для одной монеты и всех её таймфреймов.

    groups — список (df, находки, это_история[, таймфрейм]): прошлые случаи рисуются
    цветом исхода, текущие находки — синим. Если объектов больше лимита Pine (500),
    оставляем все текущие и самые свежие прошлые."""
    groups = _normalize(groups, interval)
    intervals = sorted({g[3] for g in groups}, key=lambda i: INTERVAL_SECONDS[i])
    multi = len(intervals) > 1
    objects = []                       # (таймфрейм, вид, текущий?, время, строка)
    for df, items, history, tf in groups:
        tag = tf.upper() if multi else ""
        for item in items:
            when = _end_time(df, item)
            for kind, code_lines in zip(("line", "box", "label"), _draw_item(df, item, history, tag)):
                objects += [(tf, kind, not history, when, line) for line in code_lines]

    kept = []
    for kind in ("line", "box", "label"):
        same = [o for o in objects if o[1] == kind]
        same.sort(key=lambda o: (o[2], o[3]), reverse=True)   # сначала текущие, потом свежие прошлые
        kept += same[:MAX_OBJECTS]
    kept_set = set(map(id, kept))

    title = " ".join(i.upper() for i in intervals) if intervals else interval.upper()
    header = [
        "//@version=5",
        f'indicator("Pattern Finder v{VERSION} · {symbol} {title}", overlay=true, '
        "max_lines_count=500, max_boxes_count=500, max_labels_count=500)",
    ]
    if multi:
        header.append('only_higher = input.bool(true, "Показывать только ТФ не ниже графика")')
        header += [f'show_{tf} = input.bool(true, "{tf.upper()}")' for tf in intervals]
        header.append("chart_seconds = timeframe.in_seconds()")
    body = ["if barstate.islast"]
    drawn = 0
    for tf in intervals:
        lines = [o[4] for o in objects if o[0] == tf and id(o) in kept_set]
        if not lines:
            continue
        drawn += len(lines)
        if multi:
            body.append(f"    if show_{tf} and (not only_higher or chart_seconds <= {INTERVAL_SECONDS[tf]})")
            body += ["        " + line for line in lines]
        else:
            body += ["    " + line for line in lines]
    if not drawn:
        body.append('    label.new(bar_index, high, "Паттерны не найдены")')
    return "\n".join(header + body) + "\n"


def pine_help(symbol, interval, groups):
    """Подпись к файлу разметки: что найдено СЕЙЧАС (по таймфреймам) и сколько прошлых случаев."""
    groups = _normalize(groups, interval)
    current = {}
    past = 0
    for _, items, history, tf in groups:
        if history:
            past += len(items)
        else:
            current.setdefault(tf, []).extend(it["name"] for it in items)
    tfs = sorted({g[3] for g in groups}, key=lambda i: INTERVAL_SECONDS[i])
    if len(tfs) > 1:
        now = "; ".join(f"{tf.upper()}: " + ", ".join(dict.fromkeys(names)) for tf, names in current.items())
    else:
        now = ", ".join(dict.fromkeys(n for names in current.values() for n in names))
    now = now or "—"
    head = (f"Разметка {symbol} — таймфреймы {', '.join(t.upper() for t in tfs)} в одном файле\n"
            f"Сейчас найдено: {now} (синие подписи «СЕЙЧАС»).\n"
            f"Плюс {plural(past, 'прошлый случай', 'прошлых случая', 'прошлых случаев')} твоих паттернов "
            "за 1000 свечей каждого ТФ — подписаны «был» и датой, их нет в алерте.\n")
    if len(tfs) > 1:
        head += ("На графике видны ТФ не ниже открытого (на 1H — всё, на 1D — только 1D). "
                 "Включить/выключить ТФ: шестерёнка индикатора → Inputs.\n")
    return head + "\n" + PINE_HELP.format(symbol=symbol)


PINE_HELP = (
    "Цвета: прошлые фигуры — тонкие линии, подпись зелёная/красная (цена потом выросла/упала);\n"
    "синие точки — касания линий у текущих фигур;\n"
    "зоны FVG: яркие — ещё открыты, бледные — уже закрыты;\n"
    "пунктир BOS/CHoCH — пробитый уровень; оранжевые точки EQH/EQL — ликвидность "
    "($$$ — ещё не снята);\n"
    "OB — сплошная рамка, Breaker/Mitigation — пунктир; бледные — уже пробиты.\n\n"
    "Как включить в TradingView:\n"
    "1. Открой график {symbol} (биржа Binance)\n"
    "2. Внизу открой Pine Editor\n"
    "3. Удали всё в редакторе и вставь код из файла\n"
    "4. Нажми «Add to chart»"
)
