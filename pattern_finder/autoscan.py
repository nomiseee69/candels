"""Автоскан: бот сам проверяет рынок после закрытия каждой свечи
и присылает только НОВЫЕ находки тем, кто включил автоскан."""

import time

from scanner import alert_caption, scan_user
from market import get_candles
from storage import add_sent, autoscan_users, get_sent

INTERVAL_SECONDS = {"1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400, "1w": 604800}
WEEK_OFFSET = 4 * 86400   # 01.01.1970 — четверг, а недельные свечи Binance начинаются в понедельник
DELAY = 60                # ждём минуту после закрытия свечи, чтобы биржа точно её закрыла
SAME_PATTERN_OVERLAP = 0.5


def next_run(interval, now):
    """Когда проверять в следующий раз: закрытие ближайшей свечи + DELAY (в секундах UTC)."""
    step = INTERVAL_SECONDS[interval]
    offset = WEEK_OFFSET if interval == "1w" else 0
    k = (now - offset - DELAY) // step + 1
    return offset + k * step + DELAY


class Schedule:
    """Расписание по всем таймфреймам: due() отдаёт те, у которых только что закрылась свеча."""

    def __init__(self, now):
        self.next = {i: next_run(i, now) for i in INTERVAL_SECONDS}

    def due(self, now):
        ready = [i for i, t in self.next.items() if now >= t]
        for i in ready:
            self.next[i] = next_run(i, now)
        return ready


# --- «это уже присылали?» ---------------------------------------------------------------------------
def sent_entry(card, item, now):
    df = card["df"]
    start = df["open_time"].iloc[item["start"]]
    end = df["open_time"].iloc[min(item["start"] + item["window"] - 1, len(df) - 1)]
    return {
        "symbol": card["symbol"], "interval": card["interval"], "name": item["name"],
        "start": start.timestamp(), "end": end.timestamp(), "at": now,
    }


def is_new(entry, sent):
    """Новая находка, если раньше не присылали тот же паттерн на той же монете и ТФ
    в том же месте графика (участки перекрываются больше чем наполовину)."""
    for s in sent:
        if (s["symbol"], s["interval"], s["name"]) != (entry["symbol"], entry["interval"], entry["name"]):
            continue
        overlap = min(s["end"], entry["end"]) - max(s["start"], entry["start"])
        shorter = max(min(s["end"] - s["start"], entry["end"] - entry["start"]), 1)
        if overlap >= SAME_PATTERN_OVERLAP * shorter:
            return False
    return True


def cached_fetch():
    """Загрузка свечей с кэшем: если автоскан включён у 50 человек, биржу спрашиваем один раз."""
    cache = {}

    def fetch(symbol, interval, limit):
        key = (symbol, interval, limit)
        if key not in cache:
            cache[key] = get_candles(symbol, interval, limit)
        return cache[key]
    return fetch


def run_autoscan(interval, send_cards, send_message, now=None):
    """Проверяет таймфрейм interval для всех подписчиков. Возвращает число отправленных карточек."""
    now = now or time.time()
    fetch = cached_fetch()
    total = 0
    for chat_id, patterns, intervals in autoscan_users():
        if interval not in intervals:
            continue
        cards, _ = scan_user(patterns, [interval], fetch)
        sent = get_sent(chat_id)
        new_cards, new_entries = [], []
        for card in cards:
            entries = [(item, sent_entry(card, item, now)) for item in card["items"]]
            fresh = [(item, e) for item, e in entries if is_new(e, sent)]
            if not fresh:
                continue
            items = [item for item, _ in fresh]
            new_cards.append({**card, "items": items,
                              "caption": alert_caption(card["symbol"], interval, card["df"], items)})
            new_entries += [e for _, e in fresh]
        if new_cards:
            send_message(f"🔔 Автоскан {interval.upper()}: новые находки", chat_id)
            send_cards(chat_id, new_cards)
            add_sent(chat_id, new_entries, now)
            total += len(new_cards)
    return total
