"""Совпадение по таймфреймам и поиск монеты."""

import pandas as pd

import scanner
from confluence import caption_block, confluences, score


def frame_df(n=60, freq="4h"):
    times = pd.date_range("2026-10-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})


def zone(name, kind, bottom, top):
    item = {"name": name, "score": 0.9, "start": 50, "window": 10, "kind": kind, "zone": (bottom, top)}
    if name != "FVG":
        item["family"] = "ob"
    return item


def coin(frames, trends=None):
    return {"symbol": "SOLUSDT", "history": False,
            "frames": [{"interval": tf, "df": frame_df(), "items": items, "history": False,
                        "trends": trends or {}} for tf, items in frames]}


def test_zone_inside_higher_zone_and_trend():
    c = coin([("1h", [zone("FVG", "bullish_fvg", 100.2, 100.6)]),
              ("4h", [zone("Order Block", "bullish_ob", 100.0, 101.0)])],
             trends={"1h": "up", "4h": "up", "1d": "up"})
    found = confluences(c)
    texts = [f["text"] for f in found]
    assert "1H FVG ↑ внутри 4H Order Block ↑" in texts
    assert "1H FVG ↑ по тренду старших ТФ (4H ↑, 1D ↑)" in texts
    assert found[0]["weight"] == 2 and score(found) >= 3


def test_opposite_directions_do_not_match_and_warn():
    c = coin([("1h", [zone("FVG", "bearish_fvg", 100.2, 100.6)]),
              ("4h", [zone("Order Block", "bullish_ob", 100.0, 101.0)])],
             trends={"4h": "up", "1d": "up"})
    found = confluences(c)
    assert not any("внутри" in f["text"] for f in found)
    assert any(f["weight"] < 0 and "против тренда" in f["text"] for f in found)
    assert caption_block(found).startswith("Совпадение ТФ:") or caption_block(found).startswith("⚠️")


def test_same_signal_on_several_timeframes():
    bos = lambda: {"name": "BOS", "score": 0.9, "start": 40, "window": 10, "kind": "bos",
                   "direction": "up", "level": 100.0}
    found = confluences(coin([("1h", [bos()]), ("4h", [bos()])]))
    assert {"text": "BOS ↑ сразу на 1H, 4H", "weight": 1} in found


def test_confluence_coins_go_first():
    plain = {"symbol": "AAAUSDT", "interval": "1h", "df": frame_df(), "history": False, "caption": "x",
             "items": [zone("FVG", "bullish_fvg", 1, 2), zone("Order Block", "bearish_ob", 3, 4)]}
    linked = [{"symbol": "SOLUSDT", "interval": tf, "df": frame_df(), "history": False, "caption": "y\nlink",
               "items": [zone("FVG", "bullish_fvg", 100.2, 100.6)]} for tf in ("1h", "4h")]
    coins = scanner.group_by_coin([plain] + linked)
    assert coins[0]["symbol"] == "SOLUSDT" and score(coins[0]["confluence"]) > 0


def test_scan_records_trends_of_all_timeframes():
    original = scanner.build_detectors
    candles = [[int(t.timestamp() * 1000), 1, 2, 0.5, 1.5, 1] for t in
               pd.date_range("2026-01-01", periods=150, freq="h", tz="UTC")]
    scanner.build_detectors = lambda patterns, settings=None: (
        [("FVG", lambda df: {"score": 0.9, "start": 140, "window": 5})], [])
    try:
        cards, _ = scanner.scan_user([{"name": "FVG"}], ["1h", "4h"], lambda s, i, n: candles, ["A"])
    finally:
        scanner.build_detectors = original
    assert set(cards[0]["trends"]) == {"1h", "4h"}


# --- поиск монеты ----------------------------------------------------------------------------------
def test_find_coin_in_results_by_prefix_or_live(tmp_path):
    import bot
    import storage
    original_file = storage.FILE
    storage.FILE = str(tmp_path / "users.json")          # не трогать настоящий users.json
    sent, messages, jobs = [], [], []
    saved = bot.send_card, bot.send_message, bot.start_job, bot.get_candles
    bot.send_card = lambda chat_id, card_id, c: sent.append((card_id, c["symbol"]))
    bot.send_message = lambda text, chat_id, keyboard=None: messages.append((text, keyboard))
    bot.start_job = lambda chat_id, func, *args: jobs.append((func.__name__, args))
    try:
        with bot.cards_lock:
            bot.cards_by_chat[55] = {}
        bot.remember_cards(55, [{"symbol": s, "history": False, "frames": []}
                                for s in ("SOLUSDT", "SOLVUSDT", "WLDUSDT")])
        ids = list(bot.cards_by_chat[55])
        bot.find_coin(55, "wld")                                   # точное совпадение
        assert sent[-1] == (ids[2], "WLDUSDT")
        bot.find_coin(55, "SO")                                    # несколько похожих — кнопки
        buttons = [b["text"] for row in messages[-1][1]["inline_keyboard"] for b in row
                   if b["callback_data"].startswith("v:")]
        assert set(buttons) == {"SOL", "SOLV"}
        bot.find_coin(55, "PEPE")                                  # нет в результатах — проверить сейчас
        assert jobs[-1] == ("check_coin", ("PEPEUSDT",))
        bot.get_candles = lambda *a: None                          # такой пары на Binance нет
        original_patterns = bot.get_patterns
        bot.get_patterns = lambda chat_id: [{"name": "FVG"}]
        try:
            bot.check_coin(55, "NOPEUSDT")
        finally:
            bot.get_patterns = original_patterns
        assert messages[-1][0].startswith("Не нашёл NOPEUSDT на Binance")
        bot.handle_message({"chat": {"id": 55}, "text": "sol/usdt"})
        assert sent[-1] == (ids[0], "SOLUSDT")                     # тикер текстом — тоже поиск
    finally:
        bot.send_card, bot.send_message, bot.start_job, bot.get_candles = saved
        storage.FILE = original_file
