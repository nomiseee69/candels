"""Картинки графиков для алертов и отладки."""

import matplotlib

matplotlib.use("Agg")           # без окон: бот только сохраняет картинки
import matplotlib.pyplot as plt


def plot_close(df, symbol, path, minima, maxima):
    """Цена + локальные минимумы (красные) и максимумы (зелёные)."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["open_time"], df["close"], label=symbol)
    plt.plot(df["open_time"], df["low"], linewidth=0.8, label="low")
    plt.plot(df["open_time"], df["high"], linewidth=0.8, label="high")
    plt.scatter(df.loc[maxima, "open_time"], df.loc[maxima, "high"], color="green", s=60)
    plt.scatter(df.loc[minima, "open_time"], df.loc[minima, "low"], color="red", s=60)
    plt.xlabel("Время (UTC)")
    plt.ylabel("Цена")
    plt.title(f"{symbol} PRICE")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_match(df, symbol, start, window_len, score, path):
    """Цена + оранжевая полоса на месте найденного паттерна."""
    end = start + window_len - 1
    plt.figure(figsize=(12, 6))
    plt.plot(df["open_time"], df["close"], label=symbol)
    plt.axvspan(df.loc[start, "open_time"], df.loc[end, "open_time"],
                color="orange", alpha=0.3, label=f"Совпадение {score:.0%}")
    plt.title(f"{symbol}: найденный паттерн")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _draw_level(ax, df, item):
    """BOS/CHoCH — пунктир от свинга до свечи слома; EQH/EQL — точки до снятия ликвидности."""
    last = len(df) - 1
    if item.get("kind") in ("bos", "choch"):
        color = "seagreen" if item["direction"] == "up" else "crimson"
        right, style = item["end"], "--"
        text = "BOS" if item["kind"] == "bos" else "CHoCH"
    else:
        color, style = "darkorange", ":"
        right = item["swept_at"] if item.get("swept_at") is not None else last
        text = "EQH" if item["kind"] == "eqh" else "EQL"
    x0, x1 = df["open_time"].iloc[item["start"]], df["open_time"].iloc[min(right, last)]
    ax.hlines(item["level"], x0, x1, colors=color, linestyles=style, linewidth=1.5)
    ax.text(x0, item["level"], text, color=color, fontsize=8, va="bottom")


def _time_at(df, x):
    """Время свечи x; за правым краем графика — продлеваем шагом свечи (к вершине треугольника)."""
    x = int(round(x))
    if x < len(df):
        return df["open_time"].iloc[max(x, 0)]
    step = df["open_time"].iloc[-1] - df["open_time"].iloc[-2]
    return df["open_time"].iloc[-1] + step * (x - len(df) + 1)


def _draw_range(ax, df, item, alpha):
    """Боковик: серый прямоугольник и оранжевый пунктир POC (цена с наибольшим объёмом)."""
    bottom, top = item["box"]
    end = min(item.get("end", item["start"] + item["window"] - 1), len(df) - 1)
    times = df["open_time"].iloc[item["start"]:end + 1]
    ax.fill_between(times, bottom, top, color="gray", alpha=alpha, label=item.get("name", "Range"))
    if item.get("poc") is not None:
        ax.hlines(item["poc"], times.iloc[0], times.iloc[-1], colors="darkorange", linestyles="--", linewidth=1.2)


def _draw_figure(ax, df, item, colors=("crimson", "seagreen"), width=2):
    """Линии треугольника / клина / канала — каждая от своего первого касания.
    У флага и вымпела — ещё флагшток (фиолетовая линия)."""
    if item.get("pole"):
        x0, y0, x1, y1 = item["pole"]
        ax.plot([_time_at(df, x0), _time_at(df, x1)], [y0, y1], color="purple", linewidth=width + 1)
    for key, color in zip(("upper", "lower"), colors):
        x0, y0, x1, y1 = item[key]
        ax.plot([_time_at(df, x0), _time_at(df, x1)], [y0, y1], color=color, linewidth=width)
    points = item.get("touch_points") or []
    if points:                                   # касания — почему бот считает это фигурой
        ax.scatter([_time_at(df, x) for x, _ in points], [y for _, y in points],
                   s=25, color="royalblue", zorder=5)


def _zone(ax, df, item, alpha):
    bottom, top = item["zone"]
    color = "seagreen" if item.get("kind", "").startswith("bullish") else "crimson"
    end = item.get("filled_at") if item.get("filled_at") is not None else item.get("broken_at")
    times = df["open_time"].iloc[item["start"]:(end + 1 if end is not None else None)]
    ax.fill_between(times, bottom, top, color=color, alpha=alpha, label=item["name"])


def _history_on(ax, df, symbol, occurrences):
    ax.plot(df["open_time"], df["close"], label=symbol, linewidth=1)
    for occ in occurrences:
        if "zone" in occ:                        # FVG или блок
            _zone(ax, df, occ, 0.2)
            continue
        if "level" in occ:                       # BOS / CHoCH / EQH / EQL
            _draw_level(ax, df, occ)
            continue
        if occ.get("kind") == "range":
            _draw_range(ax, df, occ, 0.2)
            continue
        end = min(occ["start"] + occ["window"] - 1, len(df) - 1)
        result = occ.get("outcome")
        color = "gray" if result is None else ("seagreen" if result["change"] > 0 else "crimson")
        ax.axvspan(df["open_time"].iloc[occ["start"]], df["open_time"].iloc[end], color=color, alpha=0.25)
        if "upper" in occ:
            _draw_figure(ax, df, {**occ, "touch_points": []}, colors=("black", "black"), width=1.5)


def _findings_on(ax, df, symbol, items):
    best = max(items, key=lambda it: it["score"])
    start, end = best["start"], min(best["start"] + best["window"] - 1, len(df) - 1)
    ax.plot(df["open_time"], df["close"], label=symbol, linewidth=1.2)
    ax.axvspan(df["open_time"].iloc[start], df["open_time"].iloc[end], color="orange", alpha=0.2,
               label=best["name"])
    for item in items:
        if "level" in item:
            _draw_level(ax, df, item)
        if "upper" in item:
            _draw_figure(ax, df, item)
        if "zone" in item:                       # зона FVG или блока — от образования до сейчас
            _zone(ax, df, item, 0.25)
        if item.get("kind") == "range":
            _draw_range(ax, df, item, 0.2)
    ax.legend(loc="upper left", fontsize=8)


def plot_history(df, symbol, occurrences, path):
    """Все прошлые случаи паттерна на графике. Цвет полосы — что было после:
    зелёный — цена выросла, красный — упала, серый — ещё рано судить."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _history_on(ax, df, symbol, occurrences)
    ax.set_title(f"{symbol}: история — зелёный = цена потом выросла, красный = упала")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_findings(df, symbol, items, path):
    """Все находки на одной монете (комбинация): для каждой — её разметка.
    Оранжевая полоса — у паттерна с наибольшим совпадением, синие точки — касания линий."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _findings_on(ax, df, symbol, items)
    ax.set_title(f"{symbol}: " + " + ".join(it["name"] for it in items))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_coin(symbol, frames, path):
    """Одна монета, несколько таймфреймов — панели друг под другом, младший ТФ сверху.
    frames — [(таймфрейм, df, находки, это_история)]."""
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 3.6 * len(frames)), squeeze=False)
    for ax, (interval, df, items, history) in zip(axes[:, 0], frames):
        if history:
            _history_on(ax, df, symbol, items)
        else:
            _findings_on(ax, df, symbol, items)
        ax.set_title(f"{symbol} {interval.upper()}: " + " + ".join(dict.fromkeys(it["name"] for it in items)),
                     fontsize=10)
        # у каждой панели своя ось времени (1H — недели, 1D — годы), подписи — на всех панелях
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- пример паттерна крупно: свечи + объём -----------------------------------------------------------
EXAMPLE_BEFORE = 40        # свечей до паттерна
EXAMPLE_AFTER = 25         # и после — видно, что было дальше
EXAMPLE_MAX = 220          # не больше 220 свечей на картинке, иначе свечи сливаются


def _candles(ax, df, lo, hi):
    for i in range(lo, hi + 1):
        o, h, low, c = (float(df[k].iloc[i]) for k in ("open", "high", "low", "close"))
        color = "#089981" if c >= o else "#f23645"
        ax.vlines(i, low, h, color=color, linewidth=0.8)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o, c)), 0.7, max(abs(c - o), (h - low) * 0.02),
                                   color=color, zorder=3))


def _span(df, item):
    """Первая и последняя свеча паттерна (номера на графике)."""
    start = int(item["start"])
    end = item.get("end")
    if end is None:
        end = start + item.get("window", 1) - 1
    return start, min(int(end), len(df) - 1)


def plot_example(df, symbol, interval, item, path, title=None, dates=True):
    """Один паттерн крупно: японские свечи, объём снизу и разметка — как трейдер увидел бы
    его в TradingView. Для обучения: «вот так бот понимает Order Block / BOS / Range»."""
    n = len(df)
    start, end = _span(df, item)
    lo = max(0, start - EXAMPLE_BEFORE)
    hi = min(n - 1, end + EXAMPLE_AFTER)
    if hi - lo > EXAMPLE_MAX:
        lo = max(0, min(start - 10, hi - EXAMPLE_MAX))
        hi = min(n - 1, lo + EXAMPLE_MAX)
    has_volume = "volume" in df
    fig, axes = plt.subplots(2 if has_volume else 1, 1, figsize=(12, 7), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [4, 1]} if has_volume else None)
    ax = axes[0, 0]
    _candles(ax, df, lo, hi)
    right = hi
    kind = str(item.get("kind", ""))
    bullish = kind.startswith("bullish") or item.get("direction") == "up"
    color = "seagreen" if bullish else "crimson"

    if "zone" in item:                                       # FVG / блок — прямоугольник зоны
        bottom, top = item["zone"]
        closed = item.get("filled_at") if item.get("filled_at") is not None else item.get("broken_at")
        stop = min(closed, hi) if closed is not None else hi
        ax.add_patch(plt.Rectangle((start - 0.5, bottom), stop - start + 1, top - bottom,
                                   color=color, alpha=0.2, zorder=2))
        if kind.endswith("_fvg"):
            for k, x in enumerate(range(start, start + 3), start=1):    # три свечи FVG
                ax.text(x, float(df["high"].iloc[x]), str(k), ha="center", va="bottom", fontsize=9)
        else:
            ax.add_patch(plt.Rectangle((start - 0.5, float(df["low"].iloc[start])), 1,
                                       float(df["high"].iloc[start] - df["low"].iloc[start]),
                                       fill=False, edgecolor="black", linewidth=1.5, zorder=4))
    if kind in ("bos", "choch"):                             # уровень свинга и свеча слома
        ax.hlines(item["level"], start, item["end"], colors=color, linestyles="--", linewidth=1.5)
        ax.scatter([start], [item["level"]], s=60, color="royalblue", zorder=5)
        ax.annotate("свинг", (start, item["level"]), textcoords="offset points", xytext=(0, 8), fontsize=9)
        close = float(df["close"].iloc[item["end"]])
        ax.scatter([item["end"]], [close], s=140, marker="*", color="gold", edgecolor="black", zorder=6)
        ax.annotate("закрытие за уровнем", (item["end"], close), textcoords="offset points",
                    xytext=(6, -14 if bullish else 10), fontsize=9)
    if kind in ("eqh", "eql"):                               # равные хаи / лои
        stop = item["swept_at"] if item.get("swept_at") is not None else hi
        ax.hlines(item["level"], start, min(stop, hi), colors="darkorange", linestyles=":", linewidth=2)
        values = df["high"] if kind == "eqh" else df["low"]
        touches = [t for t in item.get("touches", []) if lo <= t <= hi]
        ax.scatter(touches, [float(values.iloc[t]) for t in touches], s=50, color="darkorange", zorder=5)
    if "upper" in item:                                      # фигуры, флаги, вымпелы
        if item.get("pole"):
            x0, y0, x1, y1 = item["pole"]
            ax.plot([x0, x1], [y0, y1], color="purple", linewidth=3)
        for key, line_color in (("upper", "crimson"), ("lower", "seagreen")):
            x0, y0, x1, y1 = item[key]
            ax.plot([x0, x1], [y0, y1], color=line_color, linewidth=2)
        points = item.get("touch_points") or []
        ax.scatter([x for x, _ in points], [y for _, y in points], s=20, color="royalblue", zorder=5)
    if kind == "range":                                      # коробка + профиль объёма справа
        bottom, top = item["box"]
        ax.add_patch(plt.Rectangle((start - 0.5, bottom), end - start + 1, top - bottom,
                                   color="gray", alpha=0.15, zorder=2))
        if item.get("poc") is not None:
            ax.hlines(item["poc"], start, end, colors="darkorange", linestyles="--", linewidth=1.5)
            ax.annotate("POC", (end, item["poc"]), textcoords="offset points", xytext=(4, 2), fontsize=9,
                        color="darkorange")
        if has_volume:
            from ranges import volume_profile
            segment = df.iloc[start:end + 1]
            profile, centers = volume_profile(segment["high"].to_numpy(), segment["low"].to_numpy(),
                                              segment["volume"].to_numpy(), bottom, top)
            if profile.max() > 0:
                width = 18
                bar = (top - bottom) / len(profile)
                for value, center in zip(profile, centers):
                    ax.add_patch(plt.Rectangle((hi + 2, center - bar / 2), width * value / profile.max(),
                                               bar * 0.9, color="steelblue", alpha=0.5))
                right = hi + 2 + width
    if "zone" not in item and "upper" not in item and "level" not in item and kind != "range":
        ax.axvspan(start - 0.5, end + 0.5, color="orange", alpha=0.15)     # паттерн по форме

    ax.set_xlim(lo - 1, right + 1)
    ax.set_title(title or f"{symbol} {interval.upper()}")
    ax.grid(alpha=0.2)
    if has_volume:
        vax = axes[1, 0]
        volumes = df["volume"].iloc[lo:hi + 1]
        colors = ["#089981" if c >= o else "#f23645"
                  for o, c in zip(df["open"].iloc[lo:hi + 1], df["close"].iloc[lo:hi + 1])]
        vax.bar(range(lo, hi + 1), volumes, color=colors, alpha=0.6)
        average = df["volume"].rolling(20, min_periods=5).mean().shift(1).iloc[lo:hi + 1]
        vax.plot(range(lo, hi + 1), average, color="black", linewidth=1, label="средний объём (20)")
        if "zone" in item and not kind.endswith("_fvg"):        # объём на свече блока и импульсе
            vax.bar(range(start, min(start + 3, hi + 1)), df["volume"].iloc[start:min(start + 3, hi + 1)],
                    color="orange", alpha=0.9)
        vax.legend(loc="upper left", fontsize=8)
        vax.grid(alpha=0.2)
    bottom_ax = axes[-1, 0]
    ticks = list(range(lo, hi + 1, max(1, (hi - lo) // 8)))
    bottom_ax.set_xticks(ticks)
    if dates:
        bottom_ax.set_xticklabels([f"{df['open_time'].iloc[t]:%d.%m %H:%M}" for t in ticks], rotation=20, fontsize=8)
    else:                                                  # свечи со скриншота: дат нет — номера свечей
        bottom_ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
        ax.set_yticklabels([])                             # и цены — это пиксели, не доллары
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
