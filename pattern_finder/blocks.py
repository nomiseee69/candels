"""Блоки Smart Money: Order Block, Breaker Block, Mitigation Block.

Всё начинается со свинга. Возьмём свинг-хай B (для свинг-лоя — всё зеркально):

  кандидат  — последняя РАСТУЩАЯ свеча перед падением от B (среди B и 3 свечей до неё).
              Её high–low — зона блока: там крупный игрок набирал позицию.

  дальше смотрим, что сделала цена:
  1) закрылась НИЖЕ прошлого свинг-лоя (слом структуры вниз)
       → кандидат стал МЕДВЕЖЬИМ ORDER BLOCK. Цена часто возвращается в него и
         отталкивается вниз.
  2) потом (или сразу) закрылась ВЫШЕ самого свинга B — движение вниз провалилось:
       • если перед этим был слом вниз или прострел прошлого лоя (ликвидность сняли)
         → БЫЧИЙ BREAKER BLOCK: бывшая зона продавца стала поддержкой;
       • если цена даже не дошла до прошлого лоя (сделала более высокий лой)
         → БЫЧИЙ MITIGATION BLOCK: тоже поддержка, но без снятия ликвидности.

Каждое событие бот «узнаёт» только на той свече, где его увидел бы трейдер, —
без подглядывания в будущее.

Блок сильнее, если импульс от него оставил FVG (имбаланс) — пометка «с FVG».

Объём: на свече блока или на импульсе сразу после неё (2 свечи) объём должен быть
ПОВЫШЕННЫМ — не меньше 1.5× среднего за 20 свечей до блока. Так крупный игрок
«оставляет след». volume_ratio — во сколько раз объём выше среднего (×1.8),
volume_high — да/нет. Фильтр включается в ⚙️ Настройках (по умолчанию включён).
"""

import numpy as np

from smc import find_all_fvg
from structure import SWING_WINDOW
from trendlines import pivot_highs, pivot_lows

CANDLE_LOOKBACK = 3     # кандидат — среди свинга и 3 свечей до него
MAX_TRACK = 1000        # следим за судьбой кандидата до конца графика (блоки живут долго)
RECENT_BLOCK = 3        # алерт: блок сложился на последних 3 свечах…
NEAR_ZONE = 0.01        # …или цена подошла к зоне ближе 1% (ретест)
MAX_AGE = 1000          # старый блок тоже важен, если он не пробит и цена к нему подошла
VOLUME_HIGH = 1.5       # объём блока ≥ 1.5× среднего за 20 свечей — «повышенный»
VOLUME_AVERAGE = 20
VOLUME_REACH = 2        # смотрим свечу блока и 2 свечи импульса после неё

BLOCK_NAMES = {
    "bullish_ob": "бычий OB", "bearish_ob": "медвежий OB",
    "bullish_breaker": "бычий Breaker", "bearish_breaker": "медвежий Breaker",
    "bullish_mitigation": "бычий Mitigation", "bearish_mitigation": "медвежий Mitigation",
}
FAMILIES = {
    "ob": ("bullish_ob", "bearish_ob"),
    "breaker": ("bullish_breaker", "bearish_breaker"),
    "mitigation": ("bullish_mitigation", "bearish_mitigation"),
}
SHORT_TEXT = {"ob": "OB", "breaker": "BB", "mitigation": "MB"}


def family_of(kind):
    return kind.split("_", 1)[1]           # "bullish_ob" → "ob"


def _first(mask, offset):
    """Номер первой свечи, где mask истинна (mask начинается со свечи offset), или None."""
    hits = np.flatnonzero(mask)
    return int(offset + hits[0]) if len(hits) else None


def _from_highs(opens, highs, lows, closes, window):
    """Кандидаты у свинг-хаёв. Возвращает блоки в «медвежьих» терминах:
    медвежий OB, а при провале — бычий breaker / mitigation.
    Поиск «первой свечи, где…» сделан на numpy — график может быть длиной 1000 свечей."""
    n = len(closes)
    opens, highs, lows, closes = (np.asarray(a, dtype=float) for a in (opens, highs, lows, closes))
    tops = pivot_highs(highs.tolist(), window)
    bottoms = pivot_lows(lows.tolist(), window)
    blocks = []
    for pos, b in enumerate(tops):
        previous = [lo for lo in bottoms if lo < b]
        if not previous:
            continue
        prev_low = previous[-1]
        level = lows[prev_low]
        known = max(b, prev_low) + window                 # когда оба свинга уже видны
        rising = [c for c in range(b, max(b - CANDLE_LOOKBACK, 0) - 1, -1) if closes[c] > opens[c]]
        candle = rising[0] if rising else b
        bottom, top = lows[candle], highs[candle]
        # новый свинг-хай после B — дальнейшее падение уже «его», не наше
        next_top = tops[pos + 1] + window if pos + 1 < len(tops) else n
        stop = min(n, b + 1 + MAX_TRACK)

        after = slice(b + 1, stop)
        fail = _first(closes[after] > highs[b], b + 1)          # падение провалилось
        until = min(next_top, fail if fail is not None else stop)
        ob = _first(closes[b + 1:until] < level, b + 1)         # слом прошлого лоя
        if ob is None and fail is not None and fail >= next_top:
            fail = None                                         # кандидат так и не стал блоком
        swept = ob is None and fail is not None and bool((lows[b + 1:fail] < level).any())

        ob_at = max(ob, known) if ob is not None else None
        fail_at = max(fail, known) if fail is not None else None
        if ob_at is not None:
            end = fail_at if fail_at is not None else n
            blocks.append({
                "kind": "bearish_ob", "start": candle, "end": ob_at, "swing": b,
                "zone": (bottom, top),
                "touched_at": _first(highs[ob_at + 1:end] >= bottom, ob_at + 1),
                "broken_at": fail_at,
            })
        if fail_at is not None:
            kind = "bullish_breaker" if (ob_at is not None or swept) else "bullish_mitigation"
            blocks.append({
                "kind": kind, "start": candle, "end": fail_at, "swing": b,
                "zone": (bottom, top),
                "touched_at": _first(lows[fail_at + 1:] <= top, fail_at + 1),
                "broken_at": _first(closes[fail_at + 1:] < bottom, fail_at + 1),
            })
    return blocks


def volume_ratios(df):
    """Во сколько раз объём каждой свечи выше среднего за 20 свечей ДО неё.
    Нет столбца объёма (тесты) — None, тогда объём не проверяем."""
    if "volume" not in df:
        return None
    volume = df["volume"].astype(float)
    average = volume.rolling(VOLUME_AVERAGE, min_periods=5).mean().shift(1)
    return (volume / average.where(average > 0)).to_numpy()


MIRROR = {"bearish_ob": "bullish_ob", "bullish_breaker": "bearish_breaker",
          "bullish_mitigation": "bearish_mitigation"}


def all_blocks(df, window=SWING_WINDOW):
    """Все блоки на графике, по времени. У каждого: зона, когда сложился (end),
    когда цена вернулась в зону (touched_at), когда зону пробили (broken_at)."""
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    blocks = _from_highs(opens, highs, lows, closes, window)
    # свинг-лои — то же самое в зеркале: переворачиваем график ценой вниз (умножаем на −1),
    # high и low меняются местами, а потом возвращаем зону обратно
    mirrored = _from_highs([-x for x in opens], [-x for x in lows], [-x for x in highs],
                           [-x for x in closes], window)
    for blk in mirrored:
        bottom, top = blk["zone"]
        blocks.append({**blk, "kind": MIRROR[blk["kind"]], "zone": (-top, -bottom)})

    ratios = volume_ratios(df)
    # имбалансы по направлению: свеча начала FVG → бычий/медвежий
    gaps = [(g["start"], g["kind"].startswith("bullish"))
            for g in find_all_fvg(df, lookback=len(df), include_filled=True)]
    last = closes[-1]
    for blk in blocks:
        bottom, top = blk["zone"]
        blk["zone"] = (float(bottom), float(top))
        blk["window"] = blk["end"] - blk["start"] + 1
        blk["distance"] = 0.0 if bottom <= last <= top else min(abs(last - bottom), abs(last - top)) / last
        blk["family"] = family_of(blk["kind"])
        # импульс, который создал блок, оставил FVG того же направления
        bullish = blk["kind"].startswith("bullish")
        if blk["family"] == "ob":
            blk["imbalance"] = any(blk["start"] <= g <= blk["end"] and up == bullish for g, up in gaps)
        else:
            blk["imbalance"] = False
        if ratios is not None:
            near = ratios[blk["start"]:blk["start"] + VOLUME_REACH + 1]
            near = near[np.isfinite(near)]
            blk["volume_ratio"] = float(near.max()) if len(near) else None
            blk["volume_high"] = blk["volume_ratio"] is not None and blk["volume_ratio"] >= VOLUME_HIGH
        # нетронутый блок сильнее, блок с FVG — ещё сильнее
        blk["score"] = (0.92 if blk["touched_at"] is None else 0.88) + (0.05 if blk["imbalance"] else 0)
        blk["label"] = describe_block(blk)
    return sorted(blocks, key=lambda blk: blk["end"])


def describe_block(blk):
    bottom, top = blk["zone"]
    if blk["broken_at"] is not None:
        state = "пробит"
    elif blk["touched_at"] is None:
        state = "ещё не тестировался"
    else:
        state = "тестировался"
    where = "цена в зоне" if blk["distance"] == 0 else f"до зоны {blk['distance']:.1%}"
    tags = (["с FVG"] if blk.get("imbalance") else []) + (
        [f"объём ×{blk['volume_ratio']:.1f}"] if blk.get("volume_ratio") is not None else [])
    tag = f" ({', '.join(tags)})" if tags else ""
    return f"{BLOCK_NAMES[blk['kind']]}{tag} {bottom:.6g}–{top:.6g}, {state}, {where}"


def find_block(df, family, volume_only=False, choose=None):
    """Действующий блок семейства family ("ob" / "breaker" / "mitigation") для алерта:
    только что сложился или цена подошла к нему (ретест). Ближайший к цене.
    volume_only — только Order Block с повышенным объёмом (у Breaker/Mitigation не проверяем)."""
    n = len(df)
    candidates = [
        blk for blk in all_blocks(df)
        if blk["family"] == family and blk["broken_at"] is None and blk["end"] >= n - MAX_AGE
        and (blk["end"] >= n - RECENT_BLOCK or blk["distance"] <= NEAR_ZONE)
        and not (volume_only and family == "ob" and blk.get("volume_high") is False)
    ]
    if not candidates:
        return None
    if choose:                                        # самый похожий на пример пользователя
        return max(candidates, key=choose)
    return min(candidates, key=lambda blk: (blk["distance"], -blk["end"]))
