"""Путь от сообщения Telegram до выбора эталона и карточки. Сеть заменена на границе."""

from collections import OrderedDict
from pathlib import Path

import pytest
from PIL import Image

import bot
import storage
from test_menus import Recorder, buttons, datas, press


SAMPLES = Path(__file__).with_name("samples")
CHAT = 921


@pytest.fixture
def chat(tmp_path, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr(bot, "photo_queries", OrderedDict())
    monkeypatch.setattr(bot, "start_job", lambda chat_id, func, *args: func(chat_id, *args))
    downloads, scans = [], []

    def download(filename, path):
        downloads.append(filename)
        with Image.open(SAMPLES / filename) as img:
            img.convert("RGB").save(path, quality=85)

    monkeypatch.setattr(bot, "download_photo", download)
    monkeypatch.setattr(bot, "run_scan", lambda *args: scans.append(args))
    try:
        yield rec, downloads, scans
    finally:
        rec.restore()


def upload(filename):
    bot.handle_message({"chat": {"id": CHAT}, "photo": [{"file_id": "thumbnail"}, {"file_id": filename}]})


@pytest.mark.parametrize(("filename", "kind", "field"), [
    ("falling_wedge.png", "falling_wedge", "reference"), ("bearish_fvg.png", "bearish_fvg", "fingerprint")])
def test_chat_photo_confirm_once(chat, filename, kind, field):
    rec, downloads, scans = chat
    upload(filename)
    text, kb = rec.sent[-1]
    actions = [cb for cb in datas(kb) if cb.startswith("query:")]
    assert len(actions) == 1 and "Искать похожие" in text and not scans
    press(actions[0], chat_id=CHAT + 1)
    assert not scans
    press(actions[0], chat_id=CHAT)
    press(actions[0], chat_id=CHAT)
    assert len(scans) == 1 and downloads == [filename]
    assert scans[0][1]["kind"] == kind and scans[0][1][field]
    assert not storage.get_patterns(CHAT)


def test_one_clarification_reuses_selected_zone(chat):
    rec, downloads, scans = chat
    upload("real_tv_light_table.png")
    text, kb = rec.sent[-1]
    options = [b for b in buttons(kb) if b["callback_data"].startswith("query:")]
    assert text.count("?") == 1 and len(options) >= 2 and not scans
    choice = next(b for b in options if "медвежий OB" in b["text"])
    press(choice["callback_data"], chat_id=CHAT)
    assert len(downloads) == len(scans) == 1
    assert scans[0][1]["name"] == "Order Block" and not scans[0][1]["fingerprint"]["bullish"]


def test_rejection_invalidates_previous_photo(chat):
    rec, _, scans = chat
    upload("falling_wedge.png")
    old = datas(rec.sent[-1][1])[0]
    upload("cluttered_structure.png")
    reason, kb = rec.sent[-1]
    assert "несколько уровней" in reason and datas(kb) == ["tutorial:0", "menu"]
    press(old, chat_id=CHAT)
    assert not scans and not storage.get_patterns(CHAT)


def test_sections_and_tutorial(chat, monkeypatch):
    rec, _, _ = chat
    bot.handle_message({"text": "/start", "chat": {"id": CHAT}})
    assert rec.sent[-1][0].startswith(bot.WELCOME)
    assert "📖 Как сделать скриншот" in [b["text"] for b in buttons(rec.sent[-1][1])]
    press("profile", chat_id=CHAT)
    assert {"settings", "learn", "sub", "help"} <= set(datas(rec.edited[-1][1]))
    press("auto_section", chat_id=CHAT)
    assert not storage.get_autoscan(CHAT) and "auto" in datas(rec.edited[-1][1])
    press("shot:0", chat_id=CHAT)
    press("search", chat_id=CHAT)
    assert CHAT not in bot.waiting
    assert {"scan", "find", "history", "tutorial:0"} <= set(datas(rec.edited[-1][1]))
    pages = []
    def photo(path, caption, *_):
        assert Path(path).is_file()
        pages.append((Path(path).name, caption))
        return 500
    monkeypatch.setattr(bot, "send_photo", photo)
    monkeypatch.setattr(bot, "edit_photo", lambda chat_id, mid, *args: photo(*args))
    for i in range(3):
        press(f"tutorial:{i}", chat_id=CHAT, photo=i > 0)
    assert [p for p, _ in pages] == ["falling_wedge.png", "bearish_fvg.png", "cluttered_structure.png"]
    assert [text[0] for _, text in pages] == ["✅", "✅", "❌"]


def test_scan_uses_only_query_and_explains_result(monkeypatch, tmp_path):
    from test_cards import df_with_times
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    storage.add_pattern(CHAT, "BOS", None, builtin=True)
    original_settings = storage.get_settings(CHAT)
    query = {"name": "FVG", "kind": "bearish_fvg", "fingerprint": {"family": "fvg"}}
    df = df_with_times()
    def scan(patterns, intervals, symbols, settings, judge):
        assert patterns == [query] and settings["similar_only"] and not settings["combos_only"]
        cards = []
        for symbol, kind, similarity in [("SOLUSDT", "bearish_fvg", .81), ("ETHUSDT", "bullish_fvg", .95),
                                         ("BTCUSDT", "bearish_fvg", .30)]:
            item = {"name": "FVG", "kind": kind, "zone": (99, 101), "similarity": similarity, "score": .95,
                    "start": 4, "window": 3}
            cards.append(dict(symbol=symbol, interval="4h", df=df, items=[item], history=False, caption=""))
        return cards, []
    monkeypatch.setattr(bot, "scan_user", scan)
    monkeypatch.setattr(bot, "top_symbols", lambda n: ["SOLUSDT", "ETHUSDT", "BTCUSDT"])
    monkeypatch.setattr(bot, "cards_by_chat", {})
    monkeypatch.chdir(tmp_path)
    rec = Recorder()
    def send_chart(path, caption, chat_id, keyboard):
        with Image.open(path) as image:
            assert image.format == "PNG" and image.width > 100 and image.height > 100
        rec.photos.append((caption, keyboard))
        return 600
    bot.send_photo = send_chart
    try:
        bot.run_scan(CHAT, query, 77)
        assert len(rec.photos) == 1
        (cid, coin), = bot.cards_by_chat[CHAT].items()
        caption, kb = rec.photos[0]
        assert all(part in caption for part in ("SOLUSDT", "4H", "81%", "между тенями"))
        assert len(rec.edited) >= 2 and not rec.sent
        assert {"search", f"vote:{cid}:1", f"vote:{cid}:-1"} <= set(datas(kb))
        assert any("tradingview.com" in b.get("url", "") for b in buttons(kb))
        del coin["frames"][0]["items"][0]["similarity"]
        assert "%" not in bot.viewer_caption(coin, cid)
        assert storage.get_settings(CHAT) == original_settings
    finally:
        rec.restore()


def test_one_off_scan_respects_free_pattern_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FILE", str(tmp_path / "users.json"))
    bot.billing.revoke(CHAT)
    for name in ("FVG", "Клин"):
        storage.add_pattern(CHAT, name, None, builtin=True)
    walls = []
    monkeypatch.setattr(bot, "paywall", lambda chat, reason: walls.append(reason))
    quota = bot.billing.scans_left(CHAT)
    bot.run_scan(CHAT, {"name": "Канал", "kind": "channel_down"})
    assert walls == ["patterns"] and bot.billing.scans_left(CHAT) == quota
