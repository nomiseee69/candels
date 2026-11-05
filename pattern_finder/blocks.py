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

Импульс (displacement): Order Block считается, только если цена УШЛА от него резко —
за 3 свечи после блока на 1.5 средней свечи (ATR) за его границу, или оставила FVG.
Медленный уход — это не след крупного игрока.

Отработка (mitigation): блок «живой», пока цена к нему не вернулась. ПЕРВОЕ возвращение
в зону — тот самый вход, ради которого блок отмечают. После него блок отработан: бот
его больше не присылает и не рисует (кроме момента самого первого ретеста).

ЭТАЛОННЫЙ БЛОК (как его определяет трейдер; обн. 34) — для OB и Breaker обязательны все три:
  • ликвидность — свеча блока (или свинг) сняла прошлый свинг-хай / лой (собрала стопы);
  • поглощение — сразу следующая свеча — резкая ПРОТИВОПОЛОЖНАЯ, закрывается
    за тенью свечи блока (у медвежьего OB — ниже её low);
  • объём — повышенный на свече блока или на этой резкой свече.
Breaker — эталонный OB, который потом пробили. Mitigation — без снятия ликвидности
(так он определён), но с поглощением. Поля: swept, engulfed, textbook.

Объём: на свече блока или на импульсе сразу после неё (2 свечи) объём должен быть
ПОВЫШЕННЫМ — не меньше 1.5× среднего за 20 свечей до блока. Так крупный игрок
«оставляет след». volume_ratio — во сколько раз объём выше среднего (×1.8),
volume_high — да/нет. Фильтр включается в ⚙️ Настройках (по умолчанию включён).
"""

import numpy as np

from texts import where_text

from smc import find_all_fvg
from structure import SWING_WINDOW, atr
from trendlines import pivot_highs, pivot_lows

CANDLE_LOOKBACK = 3     # кандидат — среди свинга и 3 свечей до него
MAX_TRACK = 1000        # следим за судьбой кандидата до конца графика (блоки живут долго)
RECENT_BLOCK = 3        # алерт: блок сложился на последних 3 свечах…
NEAR_ZONE = 0.01        # …или цена подошла к зоне ближе 1% (ретест)
MAX_AGE = 1000          # старый блок тоже важен, если он не пробит и цена к нему подошла
VOLUME_HIGH = 1.5       # объём блока ≥ 1.5× среднего за 20 свечей — «повышенный»
VOLUME_AVERAGE = 20
VOLUME_REACH = 2        # смотрим свечу блока и 2 свечи импульса после неё
DISPLACEMENT_BARS = 3   # импульс — за 3 свечи после блока…
DISPLACEMENT_ATR = 1.5  # …цена ушла за границу блока хотя бы на 1.5 ATR (или оставила FVG)

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
    sizes = atr(df).to_numpy()
    peaks = np.array(pivot_highs(highs, window), dtype=int)
    troughs = np.array(pivot_lows(lows, window), dtype=int)
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
            c = blk["start"]
            after = slice(c + 1, min(c + 1 + DISPLACEMENT_BARS, len(closes)))
            move = (max(highs[after], default=top) - top) if bullish else (bottom - min(lows[after], default=bottom))
            blk["displacement"] = float(move / sizes[c]) if sizes[c] > 0 else 0.0
            blk["strong"] = bool(blk["imbalance"] or blk["displacement"] >= DISPLACEMENT_ATR)
        else:
            blk["imbalance"] = False
            blk["strong"] = True
        if ratios is not None:
            near = ratios[blk["start"]:blk["start"] + VOLUME_REACH + 1]
            near = near[np.isfinite(near)]
            blk["volume_ratio"] = float(near.max()) if len(near) else None
            blk["volume_high"] = blk["volume_ratio"] is not None and blk["volume_ratio"] >= VOLUME_HIGH
        _textbook(blk, highs, lows, closes, ratios, peaks, troughs)
        # нетронутый блок сильнее, блок с FVG — ещё сильнее
        blk["score"] = (0.92 if blk["touched_at"] is None else 0.88) + (0.05 if blk["imbalance"] else 0)
        blk["label"] = describe_block(blk)
    return sorted(blocks, key=lambda blk: blk["end"])


FROM_HIGH = ("bearish_ob", "bullish_breaker", "bullish_mitigation")   # блоки от свинг-хая
LIQUIDITY_LOOKBACK = 60     # прошлый свинг, который снимает блок, — не дальше 60 свечей
ENGULF_BARS = 1             # резкая противоположная свеча — СРАЗУ следующая после блока


def _textbook(blk, highs, lows, closes, ratios, peaks, troughs):
    """Эталонный ли блок: сняли ликвидность, поглотили, объём (см. описание модуля)."""
    c = int(blk["start"])
    swing = int(blk.get("swing", c))
    n = len(closes)
    top_side = blk["kind"] in FROM_HIGH                      # блок у вершины: свеча вверх, потом вниз
    first, last = min(c, swing), max(c, swing)
    if top_side:
        prior = peaks[(peaks < first) & (peaks >= first - LIQUIDITY_LOOKBACK)]
        blk["swept"] = bool(len(prior)) and max(highs[first:last + 1]) > highs[prior[-1]]
        blk["engulfed"] = any(closes[i] < lows[c] for i in range(c + 1, min(n, c + 1 + ENGULF_BARS)))
    else:
        prior = troughs[(troughs < first) & (troughs >= first - LIQUIDITY_LOOKBACK)]
        blk["swept"] = bool(len(prior)) and min(lows[first:last + 1]) < lows[prior[-1]]
        blk["engulfed"] = any(closes[i] > highs[c] for i in range(c + 1, min(n, c + 1 + ENGULF_BARS)))
    if ratios is not None:
        near = ratios[c:min(n, c + 2)]                         # объём на свече блока или следующей
        near = near[np.isfinite(near)]
        loud = bool(len(near)) and float(near.max()) >= VOLUME_HIGH
    else:
        loud = True                                            # объёма нет (скриншот) — не проверяем
    blk["loud"] = loud
    if blk["family"] == "mitigation":
        blk["textbook"] = bool(blk["engulfed"] and loud)
    else:
        blk["textbook"] = bool(blk["swept"] and blk["engulfed"] and loud)


def describe_block(blk):
    bottom, top = blk["zone"]
    if blk["broken_at"] is not None:
        state = "пробит"
    elif blk["touched_at"] is None:
        state = "ещё не тестировался"
    else:
        state = "тестировался (отработан)"
    where = where_text(blk["distance"])
    tags = (["с FVG"] if blk.get("imbalance") else []) + (
        ["повышенный объём"] if blk.get("volume_high") else [])
    tag = f" ({', '.join(tags)})" if tags else ""
    return f"{BLOCK_NAMES[blk['kind']]}{tag} {bottom:.6g}–{top:.6g}, {state}, {where}"


def is_mitigated(blk):
    """Блок отработан: цена уже возвращалась в зону (или пробила её)."""
    return blk["broken_at"] is not None or blk["touched_at"] is not None


def find_block(df, family, volume_only=False, choose=None, strict=True):
    """Блок семейства family ("ob" / "breaker" / "mitigation") для алерта:
      • только что сложился и цена к нему ещё не возвращалась;
      • или цена подошла к нему впервые (ближе 1%);
      • или ПЕРВЫЙ ретест прямо сейчас (на последних 3 свечах).
    Блок, в который цена уже заходила раньше, отработан — не присылаем.
    Order Block — только с резким импульсом (FVG или ≥ 1.5 ATR) и, если volume_only, с объёмом."""
    n = len(df)
    candidates = []
    for blk in all_blocks(df):
        if blk["family"] != family or blk["broken_at"] is not None or blk["end"] < n - MAX_AGE:
            continue
        touched = blk["touched_at"]
        first_retest = touched is not None and touched >= n - RECENT_BLOCK
        untouched = touched is None and (blk["end"] >= n - RECENT_BLOCK or blk["distance"] <= NEAR_ZONE)
        if not (first_retest or untouched):
            continue
        if family == "ob" and not blk.get("strong", True):
            continue
        if strict and not blk.get("textbook", True):
            continue                                   # не эталонный: нет ликвидности / поглощения / объёма
        if volume_only and family == "ob" and blk.get("volume_high") is False:
            continue
        if first_retest:
            blk["label"] = blk["label"].replace("тестировался (отработан)", "первый ретест сейчас")
        candidates.append(blk)
    if not candidates:
        return None
    if choose:                                        # самый похожий на пример пользователя
        return max(candidates, key=choose)
    return min(candidates, key=lambda blk: (blk["distance"], -blk["end"]))
