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


def plot_history(df, symbol, occurrences, path):
    """Все прошлые случаи паттерна на графике. Цвет полосы — что было после:
    зелёный — цена выросла, красный — упала, серый — ещё рано судить."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["open_time"], df["close"], label=symbol, linewidth=1)
    for occ in occurrences:
        if "zone" in occ:                    # открытый FVG — зона до конца графика
            bottom, top = occ["zone"]
            color = "seagreen" if occ.get("kind") == "bullish_fvg" else "crimson"
            plt.fill_between(df["open_time"].iloc[occ["start"]:], bottom, top, color=color, alpha=0.2)
            continue
        end = min(occ["start"] + occ["window"] - 1, len(df) - 1)
        result = occ.get("outcome")
        color = "gray" if result is None else ("seagreen" if result["change"] > 0 else "crimson")
        plt.axvspan(df.loc[occ["start"], "open_time"], df.loc[end, "open_time"], color=color, alpha=0.25)
        for key in ("upper", "lower"):       # линии треугольника / клина / канала
            if key in occ:
                x0, y0, x1, y1 = occ[key]
                plt.plot([df.loc[int(x0), "open_time"], df.loc[int(x1), "open_time"]], [y0, y1],
                         color="black", linewidth=1.5)
    plt.title(f"{symbol}: история — зелёный = цена потом выросла, красный = упала")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_findings(df, symbol, items, path):
    """Все находки на одной монете (комбинация): для каждой — её разметка.
    Оранжевая полоса — у паттерна с наибольшим совпадением."""
    best = max(items, key=lambda it: it["score"])
    start, end = best["start"], best["start"] + best["window"] - 1
    plt.figure(figsize=(12, 6))
    plt.plot(df["open_time"], df["close"], label=symbol, linewidth=1.2)
    plt.axvspan(df.loc[start, "open_time"], df.loc[end, "open_time"],
                color="orange", alpha=0.2, label=f"{best['name']} {best['score']:.0%}")
    for item in items:
        for key, color in (("upper", "crimson"), ("lower", "seagreen")):
            if key in item:
                x0, y0, x1, y1 = item[key]
                plt.plot([df.loc[int(x0), "open_time"], df.loc[int(x1), "open_time"]], [y0, y1],
                         color=color, linewidth=2)
        if "zone" in item:                   # зона FVG — от образования до сейчас
            bottom, top = item["zone"]
            color = "seagreen" if item.get("kind") == "bullish_fvg" else "crimson"
            times = df["open_time"].iloc[item["start"]:]
            plt.fill_between(times, bottom, top, color=color, alpha=0.25, label=item["name"])
    plt.title(f"{symbol}: " + " + ".join(it["name"] for it in items))
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
