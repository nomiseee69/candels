"""Интерфейс: меню и экраны листаются на месте, «📂 Мои паттерны» — все паттерны и галерея
примеров, результаты — один список + один просмотрщик, и отовсюду можно вернуться в меню."""

import pandas as pd

import bot
import storage
from catalog import ALL_PATTERNS


class Recorder:
    """Подменяет отправку в Telegram и запоминает, что бот «отправил»."""

    NAMES = ("send_message", "edit_message", "answer_callback", "send_document", "send_action",
             "send_photo", "edit_photo", "delete_message", "edit_keyboard", "background")

    def __init__(self):
        self.sent, self.edited, self.docs, self.photos, self.deleted, self.keyboards = [], [], [], [], [], []
        self.saved = {name: getattr(bot, name) for name in self.NAMES}
        bot.send_message = lambda text, chat_id, keyboard=None, reply_to=None: self.sent.append((text, keyboard)) or 500
        bot.edit_message = lambda chat_id, message_id, text, keyboard=None: self.edited.append((text, keyboard)) or True
        bot.answer_callback = lambda *a: None
        bot.send_action = lambda *a: None
        bot.send_document = lambda path, caption, chat_id, filename, keyboard=None, reply_to=None: \
            self.docs.append((caption, filename, keyboard, reply_to))
        bot.send_photo = lambda path, caption, chat_id, keyboard=None: self.photos.append(("new", caption, keyboard)) or 600
        bot.edit_photo = lambda chat_id, message_id, path, caption, keyboard=None: \
            self.photos.append(("edit", caption, keyboard)) or True
        bot.delete_message = lambda chat_id, message_id: self.deleted.append(message_id)
        bot.edit_keyboard = lambda chat_id, message_id, keyboard: self.keyboards.append(keyboard)
        bot.background = lambda func, *args: func(*args)          # в тестах — сразу, без потока

    def restore(self):
        for name, value in self.saved.items():
            setattr(bot, name, value)


def press(data, chat_id=5, message_id=77, photo=False):
    message = {"chat": {"id": chat_id}, "message_id": message_id}
    if photo:
        message["photo"] = [{"file_id": "x"}]
    bot.handle_callback({"id": "1", "data": data, "message": message, "from": {"id": chat_id}})


def buttons(keyboard):
    return [b for row in keyboard["inline_keyboard"] for b in row]


def datas(keyboard):
    return [b.get("callback_data") for b in buttons(keyboard)]


def use_storage(tmp_path):
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    return original


def test_main_menu_is_compact():
    names = [b["text"] for b in buttons(bot.menu_keyboard(1))]
    assert not set(names) & set(ALL_PATTERNS)
    assert names == ["Мои паттерны", "Профиль"]


def test_patterns_list_shows_saved_and_catalog_pick_flow(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        storage.add_pattern(5, "Канал", "a.png")
        press("my")
        text, keyboard = rec.edited[-1]
        assert text == "Мои паттерны"
        index = ALL_PATTERNS.index("Канал")
        assert f"gal:{index}:0" in datas(keyboard)                       # сохранённый → галерея
        picks = [d for d in datas(keyboard) if d.startswith("pick:")]
        assert not picks
        press("catalog")
        assert sum(action.startswith("pick:") for action in datas(rec.edited[-1][1])[:-1]) == len(ALL_PATTERNS)
        assert "Канал" in [b["text"] for b in buttons(keyboard)]
        assert datas(keyboard)[-1] == "menu"
        fvg = ALL_PATTERNS.index("FVG")
        press(f"pick:{fvg}")                                            # ➕ — объяснение и действия
        text, keyboard = rec.edited[-1]
        assert text.startswith("🧩 FVG") and "Как ищу:" in text and 5 not in bot.waiting
        assert "Пришли скриншот своего примера" in text
        assert datas(keyboard) == [f"shot:{fvg}", f"demo:{fvg}", f"on:{fvg}", "my", "menu"]   # свой пример — первым
        press(f"shot:{fvg}")
        text, keyboard = rec.edited[-1]
        assert bot.waiting[5] == "FVG" and text.startswith("📷 Пришли скриншот паттерна «FVG»")
        assert datas(keyboard) == ["my", "menu"]
        press("my")                                                     # назад — к списку, ожидание снято
        assert 5 not in bot.waiting and rec.edited[-1][0] == "Мои паттерны"
        press("menu")
        assert rec.edited[-1][0].startswith("Finder Candles")
    finally:
        storage.FILE = original
        rec.restore()


def test_gallery_browse_and_delete(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        saved_dir, bot.PATTERNS_DIR = bot.PATTERNS_DIR, str(tmp_path)    # картинка «нет файла» — во временную папку
        first = storage.add_pattern(5, "Канал", str(tmp_path / "a.png"), kind="channel_down")
        second = storage.add_pattern(5, "Канал", str(tmp_path / "b.png"))
        index = ALL_PATTERNS.index("Канал")
        press(f"gal:{index}:0")                                         # из текстового списка
        assert rec.deleted == [77]                                      # список убрали
        kind, caption, keyboard = rec.photos[-1]
        assert kind == "new" and caption.startswith("📂 Канал — пример 1 из 2")
        assert "скрин: нисходящий канал" in caption
        assert [b["text"] for b in keyboard["inline_keyboard"][0]] == ["◀️", "▶️"]
        assert datas(keyboard)[-3:] == [f"gscan:{first}", f"gmore:{index}:0", "my"]
        press(f"gmore:{index}:0", message_id=78, photo=True)
        other = datas(rec.photos[-1][2])
        assert {f"gask:{first}", f"shot:{index}", f"demo:{index}", f"gall:{index}"} <= set(other)
        press(f"gal:{index}:1", message_id=78, photo=True)              # ▶️ — на месте
        kind, caption, _ = rec.photos[-1]
        assert kind == "edit" and caption.startswith("📂 Канал — пример 2 из 2")
        press(f"gdel:{second}", message_id=78, photo=True)             # удалили 2-й — показан 1-й
        kind, caption, keyboard = rec.photos[-1]
        assert kind == "edit" and caption.startswith("📂 Канал — пример 1 из 1")
        assert "◀️" not in [b["text"] for b in buttons(keyboard)]
        press(f"gall:{index}", message_id=78, photo=True)               # подтверждение — кнопками
        assert datas(rec.keyboards[-1]) == [f"gyes:{index}", f"gal:{index}:0"]
        press(f"gyes:{index}", message_id=78, photo=True)
        assert storage.get_patterns(5) == [] and 78 in rec.deleted
        assert rec.sent[-1][0].startswith("У тебя пока нет")
        press("menu", message_id=79, photo=True)                        # из галереи — меню вместо картинки
        assert 79 in rec.deleted and rec.sent[-1][0].startswith("Finder Candles")
    finally:
        bot.PATTERNS_DIR = saved_dir
        storage.FILE = original
        rec.restore()


def test_settings_change_in_place(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        press("settings")
        assert rec.edited[-1][0].startswith("Настройки") and not rec.sent
        press("tf")
        assert rec.edited[-1][0].startswith("⏱ На каких")
        press("tf:1d")
        assert "✅ 1D" in [b["text"] for b in buttons(rec.keyboards[-1])]
        press("tf_done")
        assert rec.edited[-1][0].startswith("Настройки")
        press("coins")
        assert "settings" in datas(rec.edited[-1][1])
        press("coins:100")
        assert storage.get_coins(5) == 100 and rec.edited[-1][0].startswith("Настройки")
        press("auto")
        assert "Автоскан включён" in rec.edited[-1][0] and "auto" in datas(rec.edited[-1][1])
        assert not rec.sent                                              # ни одного нового сообщения
    finally:
        storage.FILE = original
        rec.restore()


def test_home_button_everywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    df = pd.DataFrame({"open_time": pd.date_range("2026-10-01", periods=60, freq="4h", tz="UTC"),
                       "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    coin = {"symbol": "SOLUSDT", "history": False,
            "frames": [{"interval": "4h", "df": df, "history": False,
                        "items": [{"name": "FVG", "score": 0.9, "start": 50, "window": 3, "kind": "bullish_fvg",
                                   "zone": (99.0, 100.0), "distance": 0.01}]}]}
    assert "home" in datas(bot.markup_keyboard("SOLUSDT", ["4h"]))
    assert "home" in datas(bot.card_keyboard(3, coin))
    assert "home" in datas(bot.patterns_keyboard(5, 3, coin))
    assert "home" in datas(bot.results_view(5, 999999)[1]) or "menu" in datas(bot.results_view(5, 999999)[1])
    url = [b for b in buttons(bot.card_keyboard(3, coin)) if "url" in b]
    assert url and "interval=240" in url[0]["url"]                      # TradingView прямо в просмотрщике
    rec = Recorder()
    try:
        press("pinehelp")
        assert datas(rec.sent[-1][1]) == ["hide", "home"]
        press("find")
        assert rec.edited[-1][0].startswith("🔎 Напиши тикер") and datas(rec.edited[-1][1]) == ["menu"]
        press("home", message_id=90, photo=True)                         # под файлом/картинкой — новое меню
        assert rec.sent[-1][0].startswith("Finder Candles") and 90 not in rec.deleted
        press("hide", message_id=91)
        assert 91 in rec.deleted
    finally:
        rec.restore()


def test_viewer_vote_marks_button_and_list_moves_down(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        df = pd.DataFrame({"open_time": pd.date_range("2026-10-01", periods=60, freq="4h", tz="UTC"),
                           "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
        coins = [{"symbol": s, "history": False, "confluence": [], "caption": f"{s} подробно",
                  "frames": [{"interval": "4h", "df": df, "history": False,
                              "items": [{"name": "FVG", "score": 0.9, "start": 50, "window": 3,
                                         "kind": "bullish_fvg", "zone": (99.0, 100.0), "distance": 0.01}]}]}
                 for s in ("SOLUSDT", "WLDUSDT")]
        drawn = []
        saved_draw = bot.draw_coin
        bot.draw_coin = lambda coin, path: drawn.append(coin["symbol"])
        try:
            bot.send_cards(5, coins, "🔍 Нашлось", edit_id=300)
            assert rec.edited[-1][0].startswith("🔍 Нашлось")              # список — на месте «Сканирую…»
            kind, caption, keyboard = rec.photos[-1]
            assert kind == "new" and caption.startswith("🔔 SOLUSDT · 1/2")
            sid = coins[0]["session"][0]
            press(f"v:{coins[1]['session'][1]}", message_id=600, photo=True)    # ▶️
            kind, caption, _ = rec.photos[-1]
            assert kind == "edit" and caption.startswith("🔔 WLDUSDT · 2/2")
            press(f"v:{sid}", message_id=300)                               # из списка — тот же просмотрщик
            assert rec.photos[-1][0] == "edit" and drawn == ["SOLUSDT", "WLDUSDT", "SOLUSDT"]
            press(f"vote:{sid}:1", message_id=600, photo=True)               # 👍 — отметка на кнопке
            assert "✅ 👍" in [b["text"] for b in buttons(rec.keyboards[-1])]
            assert storage.get_feedback(5)[0]["vote"] == 1
            press(f"det:{sid}", message_id=600, photo=True)
            assert rec.sent[-1][0] == "SOLUSDT подробно" and datas(rec.sent[-1][1]) == ["hide", "home"]
            press(f"list:{sid}", message_id=600, photo=True)                # список переезжает вниз
            assert 300 in rec.deleted and 600 in rec.deleted
            assert rec.sent[-1][0].startswith("🔍 Нашлось")
        finally:
            bot.draw_coin = saved_draw
    finally:
        storage.FILE = original
        rec.restore()


def test_markup_file_replies_to_card_with_links(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    try:
        df = pd.DataFrame({"open_time": pd.date_range("2026-10-01", periods=60, freq="4h", tz="UTC"),
                           "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
        coin = {"symbol": "SOLUSDT", "frames": [{"interval": "4h"}]}
        groups = [(df, [], True, "1h"), (df, [], True, "4h")]
        bot.send_markup(5, coin, groups, "//@version=5", "SOLUSDT_patterns.pine.txt", 123)
        caption, filename, keyboard, reply_to = rec.docs[-1]
        assert reply_to == 123 and caption.startswith("📈 SOLUSDT — разметка для TradingView")
        assert len(caption) <= bot.CAPTION_LIMIT
        urls = [b["url"] for b in buttons(keyboard) if "url" in b]
        assert len(urls) == 2 and "interval=240" in urls[1]
        assert datas(keyboard)[-2:] == ["pinehelp", "home"]                 # после TradingView — в меню
    finally:
        storage.FILE = original
        rec.restore()


def test_enable_without_screenshot(tmp_path):
    """«✅ Включить поиск» без скрина: правила (FVG) и встроенный эталон (Двойное дно)."""
    import scanner
    rec = Recorder()
    original = use_storage(tmp_path)
    saved_dir, bot.PATTERNS_DIR = bot.PATTERNS_DIR, str(tmp_path)
    try:
        fvg, double = ALL_PATTERNS.index("FVG"), ALL_PATTERNS.index("Двойное дно")
        press(f"on:{fvg}")
        assert rec.edited[-1][0].startswith("✅ Ищу «FVG»")
        press(f"on:{fvg}")                                              # второй раз — без дубля
        press(f"on:{double}")
        patterns = storage.get_patterns(5)
        assert [p["name"] for p in patterns] == ["FVG", "Двойное дно"]
        assert all(p["builtin"] and p["photo"] is None for p in patterns)
        assert len(patterns[1]["shape"]) == 50                          # эталон «W»
        detectors, skipped = scanner.build_detectors(patterns)
        assert skipped == [] and [name for name, _ in detectors] == ["FVG", "Двойное дно"]
        press(f"pick:{fvg}")
        assert "✅ Ищу. Твоих примеров: 1" in rec.edited[-1][0] and f"on:{fvg}" not in datas(rec.edited[-1][1])
        press(f"gal:{double}:0")                                        # в галерее — карточка эталона
        kind, caption, _ = rec.photos[-1]
        assert "без скриншота — встроенный эталон" in caption
        assert (tmp_path / f"_card_{double}.png").exists()
    finally:
        bot.PATTERNS_DIR = saved_dir
        storage.FILE = original
        rec.restore()


def test_welcome_for_new_user_and_commands(tmp_path):
    rec = Recorder()
    original = use_storage(tmp_path)
    saved = bot.start_job, bot.find_coin
    jobs = []
    bot.start_job = lambda chat_id, func, *args: jobs.append(func.__name__)
    bot.find_coin = lambda chat_id, query: jobs.append(("find", query))
    try:
        bot.handle_message({"chat": {"id": 9}, "text": "привет"})       # первое сообщение — приветствие
        assert rec.sent[0][0].startswith(bot.WELCOME) and datas(rec.sent[0][1]) == ["my", "profile"]
        assert rec.sent[-1][0].startswith("Finder Candles")           # и меню
        count = len(rec.sent)
        bot.handle_message({"chat": {"id": 9}, "text": "привет"})       # второй раз — только меню
        assert len(rec.sent) == count + 1
        bot.handle_message({"chat": {"id": 9}, "text": "/scan"})
        bot.handle_message({"chat": {"id": 9}, "text": "/find sol"})
        assert jobs == ["run_scan", ("find", "sol")]
        bot.handle_message({"chat": {"id": 9}, "text": "/help"})
        assert rec.sent[-1][0].startswith(bot.WELCOME)
        bot.handle_message({"chat": {"id": 9}, "text": "/patterns"})
        assert rec.sent[-1][0].startswith("У тебя пока нет")
        press("help", chat_id=9)
        assert rec.edited[-1][0].startswith(bot.WELCOME)
    finally:
        bot.start_job, bot.find_coin = saved
        storage.FILE = original
        rec.restore()


def test_demo_from_market_and_save(tmp_path):
    """«👀 Пример с рынка»: бот находит реальный случай, рисует свечи, можно сохранить в примеры."""
    import numpy as np
    rec = Recorder()
    original = use_storage(tmp_path)
    saved = bot.PATTERNS_DIR, bot.top_symbols, bot.fetch_many, bot.start_job
    bot.PATTERNS_DIR = str(tmp_path)
    rng = np.random.default_rng(3)
    t0 = 1_780_000_000_000

    def candles(symbol, interval, limit):
        closes = 100 + np.cumsum(rng.normal(0, 1, 400))
        return [[t0 + i * 14_400_000, c - 0.3, c + 1, c - 1, c, 1 + (i % 7)] for i, c in enumerate(closes)]

    bot.top_symbols = lambda n: ["AAAUSDT", "BBBUSDT"]
    bot.fetch_many = lambda symbols, interval, limit, fetch: {s: candles(s, interval, limit) for s in symbols}
    bot.start_job = lambda chat_id, func, *args: func(chat_id, *args)
    bot.demo_cache.clear()
    try:
        index = ALL_PATTERNS.index("BOS")
        press(f"demo:{index}")
        kind, caption, keyboard = rec.photos[-1]
        assert kind == "new" and caption.startswith("👀 BOS — пример с рынка (1/")
        assert "⭐" in caption and f"on:{index}" in datas(keyboard)
        assert any("url" in b for b in buttons(keyboard))
        press(f"demo:{index}:1", message_id=80, photo=True)            # 🔄 — на месте
        assert rec.photos[-1][0] == "edit" and "(2/" in rec.photos[-1][1]
        channel = ALL_PATTERNS.index("Канал")
        bot.demo_shown[(5, channel)] = None
        lines = bot.demo_occurrences("Канал")
        if lines:                                                       # у фигуры — «💾 В мои примеры»
            press(f"demo:{channel}")
            assert f"dsave:{channel}" in datas(rec.photos[-1][2])
            press(f"dsave:{channel}")
            saved_examples = storage.get_patterns(5)
            assert saved_examples[-1]["name"] == "Канал" and saved_examples[-1]["reference"]
    finally:
        bot.PATTERNS_DIR, bot.top_symbols, bot.fetch_many, bot.start_job = saved
        storage.FILE = original
        rec.restore()


def test_network_errors_hide_token_and_back_off():
    import notifier
    import requests
    saved = notifier.TOKEN
    notifier.TOKEN = "123:SECRET"
    try:
        error = requests.ConnectTimeout("HTTPSConnectionPool(host='api.telegram.org'): /bot123:SECRET/getUpdates")
        assert "SECRET" not in notifier.safe_error(error) and "***" in notifier.safe_error(error)
        bot_saved = bot.safe_error
        bot.safe_error = notifier.safe_error
        try:
            hint = bot.network_hint(error, 1, 5)
        finally:
            bot.safe_error = bot_saved
        assert "SECRET" not in hint and "TELEGRAM_PROXY" in hint
    finally:
        notifier.TOKEN = saved
    assert [bot.network_wait(n) for n in (1, 2, 3, 4, 5, 9)] == [5, 10, 20, 40, 60, 60]
