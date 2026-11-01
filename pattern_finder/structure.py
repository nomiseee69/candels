"""Структура рынка (Smart Money Concepts): BOS, CHoCH, равные хаи и лои (EQH / EQL).

Свинг (swing high / low) — локальный экстремум: свеча, high которой выше SWING_WINDOW
свечей слева и справа. Узнать о свинге можно только через SWING_WINDOW свечей после него,
поэтому бот «видит» свинг ровно тогда, когда его увидел бы трейдер.

BOS (Break of Structure)  — цена ЗАКРЫЛАСЬ за последним свингом по направлению тренда:
                            тренд продолжается.
CHoCH (Change of Character) — первый слом ПРОТИВ тренда: был нисходящий, а цена закрылась
                            выше последнего свинг-хая (или наоборот). Возможная смена тренда.
EQH / EQL (equal highs / lows) — два и больше свинга почти на одной цене. Над равными
                            хаями копятся стопы продавцов — «ликвидность», к которой цена
                            часто идёт, чтобы её снять.
"""

import pandas as pd

from texts import plural
from trendlines import pivot_highs, pivot_lows

ATR_PERIOD = 14
SWING_WINDOW = 5         # свинг для структуры: экстремум среди 5 свечей с каждой стороны
LIQUIDITY_WINDOW = 3     # для равных хаёв/лоёв берём и свинги помельче
RECENT_BREAK = 3         # алерт BOS/CHoCH: слом случился на последних 3 свечах
EQ_TOLERANCE_ATR = 0.15  # «равные» = отличаются не больше чем на 15% среднего хода свечи
EQ_MIN_GAP = 5           # между касаниями хотя бы 5 свечей, иначе это одна вершина
EQ_LOOKBACK_SWINGS = 5   # с каким числом прошлых свингов сравнивать новый
RECENT_EQ = 10           # алерт EQH/EQL: уровень сложился недавно…
NEAR_LEVEL = 0.01        # …или цена подошла к нему ближе 1%

STRUCTURE_NAMES = {"bos": "BOS", "choch": "CHoCH"}
EQ_NAMES = {"eqh": "равные хаи", "eql": "равные лои"}


def atr(df, period=ATR_PERIOD):
    """Средний истинный диапазон свечи — «обычный» размер движения на этой монете."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


# --- BOS / CHoCH ------------------------------------------------------------------------------
def _break_event(kind, direction, level, swing, at, close, size):
    beyond = abs(close - level) / size if size > 0 else 0
    name = STRUCTURE_NAMES[kind]
    side = "выше" if direction == "up" else "ниже"
    label = f"{name} {'вверх' if direction == 'up' else 'вниз'}: закрытие {side} {level:.6g}"
    if kind == "choch":
        label += " — возможна смена тренда"
    return {
        "kind": kind,
        "direction": direction,
        "level": float(level),
        "start": int(swing),            # свеча-свинг, которую пробили
        "end": int(at),                 # свеча, закрывшаяся за уровнем
        "window": int(at - swing + 1),
        "score": float(min(1.0, 0.85 + 0.15 * min(1.0, beyond))),
        "label": label,
    }


def structure_events(df, window=SWING_WINDOW):
    """Все сломы структуры по порядку. Идём по свечам слева направо, как трейдер:
    помним последний подтверждённый свинг-хай и свинг-лой и ждём закрытия за ними."""
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    sizes = atr(df).tolist()
    # свинг j становится известен на свече j + window
    confirm_high = {j + window: j for j in pivot_highs(highs, window)}
    confirm_low = {j + window: j for j in pivot_lows(lows, window)}

    last_high = last_low = None
    trend = None
    events = []
    for i in range(len(df)):
        if i in confirm_high:
            last_high = confirm_high[i]
        if i in confirm_low:
            last_low = confirm_low[i]
        if last_high is not None and closes[i] > highs[last_high]:
            kind = "choch" if trend == "down" else "bos"
            events.append(_break_event(kind, "up", highs[last_high], last_high, i, closes[i], sizes[i]))
            trend, last_high = "up", None          # уровень пробит — ждём следующий свинг
        elif last_low is not None and closes[i] < lows[last_low]:
            kind = "choch" if trend == "up" else "bos"
            events.append(_break_event(kind, "down", lows[last_low], last_low, i, closes[i], sizes[i]))
            trend, last_low = "down", None
    return events


def find_break(df, kind):
    """Свежий BOS или CHoCH (kind = "bos" / "choch") на последних свечах, или None."""
    n = len(df)
    recent = [e for e in structure_events(df) if e["kind"] == kind and e["end"] >= n - RECENT_BREAK]
    return recent[-1] if recent else None


# --- EQH / EQL ------------------------------------------------------------------------------------
def _equal_side(df, side, sizes, window):
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    is_high = side == "eqh"
    values = highs if is_high else lows
    pivots = pivot_highs(highs, window) if is_high else pivot_lows(lows, window)

    groups = []          # касания одного уровня: [номера свечей]
    group_of = {}        # свеча-касание → её группа
    for pos, j in enumerate(pivots):
        tol = EQ_TOLERANCE_ATR * sizes[j]
        for k in reversed(pivots[max(0, pos - EQ_LOOKBACK_SWINGS):pos]):
            if j - k < EQ_MIN_GAP or abs(values[j] - values[k]) > tol:
                continue
            # между касаниями цена не должна была уйти за уровень — иначе ликвидность уже сняли
            between = values[k + 1:j]
            if between and (max(between) > max(values[j], values[k]) if is_high
                            else min(between) < min(values[j], values[k])):
                continue
            group = group_of.get(k)
            if group is None or group[-1] != k:
                group = [k]
                groups.append(group)
            group.append(j)
            group_of[j] = group
            break

    n = len(df)
    last = closes[-1]
    result = []
    for touches in groups:
        prices = [values[t] for t in touches]
        level = max(prices) if is_high else min(prices)
        first, end = touches[0], touches[-1]
        swept_at = next(
            (i for i in range(end + 1, n) if (highs[i] > level if is_high else lows[i] < level)), None)
        tol = EQ_TOLERANCE_ATR * sizes[end]
        spread = max(prices) - min(prices)
        distance = abs(last - level) / last
        if swept_at is None:
            status = "ликвидность не снята, " + ("цена у уровня" if distance < 0.001 else f"до уровня {distance:.1%}")
        else:
            back = closes[swept_at] < level if is_high else closes[swept_at] > level
            status = "ликвидность снята" + (" прострелом (закрылась обратно)" if back else "")
        result.append({
            "kind": side,
            "level": float(level),
            "touches": touches,
            "start": first,
            "end": end,
            "window": end - first + 1,
            "swept_at": swept_at,
            "distance": float(distance),
            "score": float(1 - 0.15 * (spread / tol if tol > 0 else 0)),
            "label": f"{EQ_NAMES[side]} {level:.6g} ({plural(len(touches), 'касание', 'касания', 'касаний')}), {status}",
        })
    return result


def equal_levels(df, window=LIQUIDITY_WINDOW):
    """Все уровни равных хаёв и лоёв за период, по времени."""
    sizes = atr(df).tolist()
    levels = _equal_side(df, "eqh", sizes, window) + _equal_side(df, "eql", sizes, window)
    return sorted(levels, key=lambda f: f["start"])


def find_equal(df):
    """Незабранная ликвидность для алерта: равные хаи/лои, которые сложились недавно
    или к которым цена подошла ближе 1%. Берём ближайший к цене уровень."""
    n = len(df)
    candidates = [
        f for f in equal_levels(df)
        if f["swept_at"] is None
        and (f["end"] >= n - RECENT_EQ - LIQUIDITY_WINDOW or f["distance"] <= NEAR_LEVEL)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda f: f["distance"])
