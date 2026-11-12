"""Совпадение по таймфреймам (confluence): когда находки разных ТФ подтверждают друг друга.

Трейдеры больше доверяют сетапу, если его видно сразу на нескольких масштабах.
Бот проверяет три вида совпадений:

  1) зона в зоне — FVG или блок младшего ТФ лежит внутри (или пересекает) зону
     старшего ТФ того же направления: «1H FVG ↑ внутри 4H Order Block ↑»;
  2) по тренду — направление находки совпадает с трендом всех старших ТФ
     (тренд = куда был последний слом структуры). Против тренда — предупреждение ⚠️;
  3) один сигнал на нескольких ТФ — например, BOS вверх и на 1H, и на 4H.

Каждое совпадение даёт очки; монеты с совпадениями поднимаются наверх и помечаются 🔗.
"""

from structure import structure_events

INTERVAL_ORDER = ["1h", "4h", "12h", "1d", "1w"]
ARROW = {"up": "↑", "down": "↓"}


def direction(item):
    """Куда «смотрит» находка: "up", "down" или None (у фигур и ликвидности не задано)."""
    kind = str(item.get("kind", ""))
    if kind.startswith("bullish"):
        return "up"
    if kind.startswith("bearish"):
        return "down"
    if kind in ("bos", "choch"):
        return item.get("direction")
    return None


def trend(df):
    """Тренд по последнему слому структуры: "up", "down" или None."""
    events = structure_events(df)
    return events[-1]["direction"] if events else None


def _rank(tf):
    return INTERVAL_ORDER.index(tf)


def _name(tf, item):
    arrow = ARROW.get(direction(item), "")
    return f"{tf.upper()} {item['name']}" + (f" {arrow}" if arrow else "")


def coin_trends(coin):
    """Тренды всех просканированных ТФ монеты. Сканер кладёт их в карточку ("trends");
    если их нет — считаем по графикам тех ТФ, где что-то нашлось."""
    trends = {}
    for frame in coin["frames"]:
        trends.update(frame.get("trends") or {})
        df = frame.get("df")
        if frame["interval"] not in trends and df is not None and {"high", "low", "close"} <= set(df.columns):
            trends[frame["interval"]] = trend(df)
    return trends


def confluences(coin):
    """Совпадения по ТФ: [{"text", "weight"}], самые сильные первыми. weight < 0 — предупреждение."""
    entries = [(f["interval"], it) for f in coin["frames"] if not f.get("history") for it in f["items"]]
    trends = coin_trends(coin)
    found = []

    # 1) зона в зоне: младший ТФ внутри старшего, то же направление
    for tf_low, low in entries:
        for tf_high, high in entries:
            if _rank(tf_low) >= _rank(tf_high) or "zone" not in low or "zone" not in high:
                continue
            if direction(low) is None or direction(low) != direction(high):
                continue
            (l0, l1), (h0, h1) = low["zone"], high["zone"]
            if max(l0, h0) > min(l1, h1):
                continue                                         # не пересекаются
            how = "внутри" if h0 <= l0 and l1 <= h1 else "пересекается с"
            found.append({"text": f"{_name(tf_low, low)} {how} {_name(tf_high, high)}", "weight": 2})

    # 2) по тренду старших ТФ (или против него)
    for tf, item in entries:
        side = direction(item)
        higher = {t: v for t, v in trends.items() if _rank(t) > _rank(tf) and v}
        if side is None or not higher:
            continue
        listed = ", ".join(f"{t.upper()} {ARROW[v]}" for t, v in sorted(higher.items(), key=lambda kv: _rank(kv[0])))
        if all(v == side for v in higher.values()):
            found.append({"text": f"{_name(tf, item)} по тренду старших ТФ ({listed})", "weight": 1})
        elif all(v != side for v in higher.values()):
            found.append({"text": f"{_name(tf, item)} против тренда старших ТФ ({listed})", "weight": -1})

    # 3) один и тот же сигнал на нескольких ТФ
    by_signal = {}
    for tf, item in entries:
        side = direction(item)
        if side is not None:
            by_signal.setdefault((item["name"], side), set()).add(tf)
    for (name, side), tfs in by_signal.items():
        if len(tfs) >= 2:
            ordered = ", ".join(t.upper() for t in sorted(tfs, key=_rank))
            found.append({"text": f"{name} {ARROW[side]} сразу на {ordered}", "weight": 1})

    return sorted(found, key=lambda c: c["weight"], reverse=True)


def score(found):
    """Сила совпадений — сумма положительных весов."""
    return sum(c["weight"] for c in found if c["weight"] > 0)


def caption_block(found):
    """Текст для подписи алерта."""
    good = [c for c in found if c["weight"] > 0]
    bad = [c for c in found if c["weight"] < 0]
    lines = []
    if good:
        lines.append("Совпадение ТФ:")
        lines += [f"• {c['text']}" for c in good]
    lines += [f"{c['text']}" for c in bad]
    return "\n".join(lines)
