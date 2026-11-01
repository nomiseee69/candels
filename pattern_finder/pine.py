"""Разметка для TradingView: скрипт Pine, который рисует найденные паттерны.

Ссылкой рисовать на чужом графике TradingView нельзя, поэтому бот генерирует
маленький индикатор. Пользователь вставляет его в Pine Editor → «Add to chart»,
и линии, зоны и подписи появляются прямо на его графике. Привязка — ко времени
свечей, поэтому разметка встаёт на место на любом таймфрейме этой монеты.

Рисуется всё сразу: прошлые случаи (цвет = что было после) и текущие находки (синие).
"""

# Pine разрешает не больше 500 объектов каждого вида на графике
MAX_OBJECTS = 450
APEX_REACH = 2.0       # линии фигуры продлеваем до вершины, но не дальше 2 ширин фигуры
EXTEND_RATIO = 0.25    # если вершины нет (канал) — продлеваем на 25% ширины


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


def _line_end(item):
    """До какой свечи вести линии фигуры: до вершины треугольника/клина
    (если она недалеко впереди), иначе — чуть дальше конца фигуры."""
    x0, u0, x1, u1 = item["upper"]
    _, l0, _, l1 = item["lower"]
    width = max(x1 - x0, 1)
    up_slope = (u1 - u0) / width
    lo_slope = (l1 - l0) / width
    if up_slope < lo_slope:                       # линии сходятся
        apex = x1 + (u1 - l1) / (lo_slope - up_slope)
        if x1 <= apex <= x1 + APEX_REACH * width:
            return apex, up_slope, lo_slope
    return x1 + EXTEND_RATIO * width, up_slope, lo_slope


def _draw_item(df, item, history):
    """Строки Pine для одной находки. Возвращает (линии, рамки, подписи)."""
    lines, boxes, labels = [], [], []
    name = item["name"].replace('"', "'")

    if "upper" in item:                                         # треугольник, клин, канал
        x0, u0, x1, u1 = item["upper"]
        _, l0, _, _ = item["lower"]
        if history:
            end, up_slope, lo_slope = x1, *(_line_end(item)[1:])
        else:
            end, up_slope, lo_slope = _line_end(item)
        color = _outcome_color(item) if history else "color.blue"
        for start_price, slope, side in ((u0, up_slope, "color.red"), (l0, lo_slope, "color.green")):
            end_price = start_price + slope * (end - x0)
            lines.append(
                f"line.new({_time(df, x0)}, {_price(start_price)}, {_time(df, end)}, {_price(end_price)}, "
                f"xloc=xloc.bar_time, color={side}, width=2)"
            )
        labels.append((x0, u0, f"{name} {item['score']:.0%}", color))

    elif "zone" in item:                                        # FVG
        bottom, top = item["zone"]
        color = "color.green" if item.get("kind") == "bullish_fvg" else "color.red"
        filled_at = item.get("filled_at")
        if filled_at is None:                                   # открыт — тянем вправо
            right, extend, alpha, text = len(df) - 1, "extend.right", 80, "FVG"
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
        labels.append((x0, top, f"{name} {item['score']:.0%}", color))

    label_lines = [
        f'label.new({_time(df, x)}, {_price(y)}, "{text}", xloc=xloc.bar_time, '
        f"style=label.style_label_down, color={color}, textcolor=color.white, size=size.small)"
        for x, y, text, color in labels
    ]
    return lines, boxes, label_lines


def pine_script(symbol, interval, groups):
    """Код индикатора Pine v5.

    groups — список (df, находки, это_история): прошлые случаи рисуются цветом исхода,
    текущие находки — синим и с продлёнными до вершины линиями."""
    lines, boxes, labels = [], [], []
    for df, items, history in groups:
        for item in items:
            l, b, t = _draw_item(df, item, history)
            lines += l
            boxes += b
            labels += t
    # свежие объекты в конце списков — при переполнении отбрасываем самые старые
    body = lines[-MAX_OBJECTS:] + boxes[-MAX_OBJECTS:] + labels[-MAX_OBJECTS:]
    header = [
        "//@version=5",
        f'indicator("Pattern Finder · {symbol} {interval}", overlay=true, '
        "max_lines_count=500, max_boxes_count=500, max_labels_count=500)",
        "if barstate.islast",
    ]
    if not body:
        body = ['label.new(bar_index, high, "Паттерны не найдены")']
    return "\n".join(header + ["    " + line for line in body]) + "\n"


PINE_HELP = (
    "Разметка всех паттернов {symbol} {interval}:\n"
    "синие — найдены сейчас; зелёные/красные — прошлые (цена потом выросла/упала);\n"
    "зоны FVG: яркие — ещё открыты, бледные — уже закрыты.\n\n"
    "Как включить в TradingView:\n"
    "1. Открой график {symbol} (биржа Binance)\n"
    "2. Внизу открой Pine Editor\n"
    "3. Удали всё в редакторе и вставь код из файла\n"
    "4. Нажми «Add to chart»"
)
