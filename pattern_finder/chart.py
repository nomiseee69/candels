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


def _draw_figure(ax, df, item, colors=("crimson", "seagreen"), width=2):
    """Линии треугольника / клина / канала — каждая от своего первого касания."""
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
