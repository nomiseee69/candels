"""Главная идея: бот ищет похожее на ТВОИ примеры — и для SMC тоже.
Скрин → свечи → твой паттерн → отпечаток → «похоже на твой пример N%» на рынке."""

import shutil

import numpy as np
import pandas as pd

import bot
import fingerprint as fpm
import scanner
import storage
import visual
from blocks import all_blocks, find_block
from screens import screenshot
from test_ranges_flags import range_df
from vision import extract_candles


def random_df(n=80, seed=4):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.roll(c, 1)
    o[0] = c[0]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + rng.uniform(0.1, 1, n),
                         "low": np.minimum(o, c) - rng.uniform(0.1, 1, n), "close": c,
                         "volume": rng.lognormal(0, 0.5, n)})


def ob_df(impulse=8.0, volume=3.0, seed=0, before=60, after=15):
    """Пила вниз со свингами → красная свеча → импульс вверх на `impulse` средних свечей
    с объёмом ×volume → откат. Средняя свеча ≈ 1."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    for i in range(before):
        closes.append(closes[-1] + (-0.6 if (i // 6) % 2 == 0 else 0.35) + rng.normal(0, 0.15))
    opens, ends, volumes = list(closes[:-1]), list(closes[1:]), list(rng.lognormal(0, 0.2, before))
    opens.append(ends[-1]); ends.append(ends[-1] - 0.8); volumes.append(volume)
    for k in range(4):
        opens.append(ends[-1]); ends.append(ends[-1] + impulse / 4); volumes.append(volume if k < 2 else 1.2)
    for k in range(after):
        opens.append(ends[-1]); ends.append(ends[-1] + (-0.3 if k < after // 2 else 0.1) + rng.normal(0, 0.1))
        volumes.append(rng.lognormal(0, 0.2))
    o, c = np.array(opens), np.array(ends)
    return pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=len(c), freq="4h", tz="UTC"),
                         "open": o, "high": np.maximum(o, c) + rng.uniform(0.1, 0.3, len(c)),
                         "low": np.minimum(o, c) - rng.uniform(0.1, 0.3, len(c)), "close": c,
                         "volume": np.array(volumes)})


# --- свечи со скрина ---------------------------------------------------------------------------
def test_candles_read_from_screenshot_dark_and_light(tmp_path):
    df = random_df()
    for theme in ("dark", "light"):
        candles = extract_candles(screenshot(df, tmp_path / f"{theme}.png", theme))
        assert candles is not None and abs(len(candles) - len(df)) <= 3 and "volume" in candles
        m = min(len(candles), len(df))
        shift = len(df) - len(candles) if len(candles) < len(df) and theme == "dark" else 0
        ref = df.iloc[shift:shift + m]
        for key in ("high", "low", "close"):
            assert np.corrcoef(candles[key][:m], ref[key])[0, 1] > 0.95, (theme, key)
        same = np.sign(candles["close"][:m].to_numpy() - candles["open"][:m].to_numpy()) == \
            np.sign(ref["close"].to_numpy() - ref["open"].to_numpy())
        assert same.mean() > 0.9                                            # цвет свечи понят верно
        assert np.corrcoef(candles["volume"][:m], ref["volume"])[0, 1] > 0.9  # объём снизу прочитан


def test_line_chart_has_no_candles(w_line):
    assert extract_candles(w_line) is None


# --- отпечаток и сравнение ------------------------------------------------------------------------
def test_order_block_fingerprint_from_screenshot(tmp_path):
    example = ob_df(impulse=8, volume=3, seed=1)
    candles = extract_candles(screenshot(example.tail(70).reset_index(drop=True), tmp_path / "ob.png"))
    fp, item = fpm.screenshot_fingerprint("Order Block", candles)
    assert fp["family"] == "block" and item["kind"] == "bullish_ob"
    assert {"zone_atr", "impulse_atr", "volume_ratio", "imbalance"} <= set(fp["features"])
    assert fp["window"]["ohlc"] and "импульс" in fpm.describe(fp)


def test_similar_block_scores_higher_than_different_one():
    example = ob_df(impulse=8, volume=3, seed=1)
    ref = [b for b in all_blocks(example) if b["kind"] == "bullish_ob"][-1]
    fp = fpm.make_fingerprint("Order Block", example, {**ref, "name": "Order Block"})

    def score(impulse, volume):
        market = ob_df(impulse=impulse, volume=volume, seed=5)
        block = [b for b in all_blocks(market) if b["kind"] == "bullish_ob"][-1]
        return fpm.similarity(fp, market, block)

    assert score(8, 3) > 0.75 and score(8, 3) > score(3, 1.6) + 0.15


def test_visual_mirror_and_identity():
    df = random_df(60)
    look = visual.look(visual.normalize_ohlc(df))
    assert visual.similarity(look, look) == 1.0
    flipped = df.copy()
    flipped[["open", "high", "low", "close"]] = 300 - df[["open", "low", "high", "close"]].to_numpy()
    mirror_look = visual.look(visual.normalize_ohlc(flipped, mirror=True))
    assert visual.similarity(look, mirror_look) > 0.95                      # медвежий = зеркальный бычий
    other = visual.look(visual.normalize_ohlc(random_df(60, seed=9)))
    assert visual.similarity(look, other) < 0.5


def test_find_block_chooses_most_similar_candidate():
    df = ob_df(impulse=8, volume=3, seed=2)
    blocks = [b for b in all_blocks(df) if b["family"] == "ob" and b["broken_at"] is None]
    assert blocks
    target = blocks[0]
    chosen = find_block(df, "ob", choose=lambda b: 1.0 if b is target else 0.0)
    if chosen is not None:
        assert chosen["zone"] == target["zone"] or chosen is target


def retest(df, steps=8):
    """Цена возвращается к бычьему OB (ретест) — тогда бот присылает его как находку."""
    block = [b for b in all_blocks(df) if b["kind"] == "bullish_ob"][-1]
    target = block["zone"][1] * 1.002
    closes = np.linspace(df["close"].iloc[-1], target, steps + 1)[1:]
    opens = np.concatenate([[df["close"].iloc[-1]], closes[:-1]])
    extra = pd.DataFrame({
        "open_time": pd.date_range(df["open_time"].iloc[-1], periods=steps + 1, freq="4h")[1:],
        "open": opens, "high": np.maximum(opens, closes) + 0.1, "low": np.minimum(opens, closes) - 0.1,
        "close": closes, "volume": 1.0})
    return pd.concat([df, extra], ignore_index=True)


def test_scanner_writes_similarity_and_filters_unlike():
    example = ob_df(impulse=8, volume=3, seed=1)
    ref = [b for b in all_blocks(example) if b["kind"] == "bullish_ob"][-1]
    fp = fpm.make_fingerprint("Order Block", example, {**ref, "name": "Order Block"})
    patterns = [{"name": "Order Block", "fingerprint": fp}]

    def raw(df):
        rows = [[int(t.timestamp() * 1000), o, h, lo, c, v] for t, o, h, lo, c, v in
                zip(df["open_time"], df["open"], df["high"], df["low"], df["close"], df["volume"])]
        return [rows[0]] * 60 + rows

    like, unlike = raw(retest(ob_df(8, 3, seed=5))), raw(retest(ob_df(3, 1.6, seed=5)))
    fetch = lambda symbol, interval, limit: like if symbol == "LIKEUSDT" else unlike
    settings = {"ob_volume": False}
    cards, _ = scanner.scan_user(patterns, ["4h"], fetch, ["LIKEUSDT", "UNLIKEUSDT"], settings)
    found = {c["symbol"]: c["items"][0] for c in cards}
    assert found and all("similarity" in it for it in found.values())
    if len(found) == 2:
        assert found["LIKEUSDT"]["similarity"] > found["UNLIKEUSDT"]["similarity"]
    assert "похоже на твой пример" in scanner.describe_item(found["LIKEUSDT"])
    strict, _ = scanner.scan_user(patterns, ["4h"], fetch, ["LIKEUSDT", "UNLIKEUSDT"],
                                  {**settings, "similar_only": True})
    assert all(it["similarity"] >= 0.6 for c in strict for it in c["items"])


def test_range_fingerprint_from_screenshot(tmp_path):
    df = range_df().tail(170).reset_index(drop=True)
    candles = extract_candles(screenshot(df, tmp_path / "range.png", width=14))
    fp, item = fpm.screenshot_fingerprint("Range", candles)
    assert fp is not None and item["kind"] == "range"
    assert 0.25 <= fp["features"]["poc_rel"] <= 0.75 and fp["features"]["touches"] >= 4


# --- бот: скрин SMC-паттерна ----------------------------------------------------------------------
class Sent:
    def __init__(self):
        self.messages, self.photos = [], []
        self.saved = bot.send_message, bot.send_photo, bot.download_photo, bot.PATTERNS_DIR
        bot.send_message = lambda text, chat_id, keyboard=None, reply_to=None: self.messages.append((text, keyboard))
        bot.send_photo = lambda path, caption, chat_id, keyboard=None: self.photos.append((caption, keyboard))

    def restore(self):
        bot.send_message, bot.send_photo, bot.download_photo, bot.PATTERNS_DIR = self.saved


def test_bot_reads_smc_example_from_screenshot(tmp_path):
    sent = Sent()
    original = storage.FILE
    storage.FILE = str(tmp_path / "users.json")
    bot.PATTERNS_DIR = str(tmp_path)
    shot = screenshot(ob_df(8, 3, seed=1).tail(70).reset_index(drop=True), tmp_path / "ob.png")
    bot.download_photo = lambda file_id, path: shutil.copy(shot, path)
    try:
        bot.waiting[5] = "Order Block"
        bot.handle_message({"chat": {"id": 5}, "photo": [{"file_id": "x"}]})
        caption, keyboard = sent.photos[-1]
        assert caption.startswith("Вот что я увидел на твоём скрине") and "импульс" in caption
        saved = storage.get_patterns(5)[-1]
        assert saved["name"] == "Order Block" and saved["fingerprint"]["family"] == "block"
        assert keyboard["inline_keyboard"][0][0]["callback_data"] == f"ok:{saved['id']}"
        # картинка без свечей — честно говорю, что паттерн не нашёл
        blank = tmp_path / "blank.png"
        screenshot(random_df(25).assign(high=lambda d: d["close"], low=lambda d: d["close"],
                                        open=lambda d: d["close"]), blank)
        bot.download_photo = lambda file_id, path: shutil.copy(blank, path)
        bot.waiting[5] = "FVG"
        bot.handle_message({"chat": {"id": 5}, "photo": [{"file_id": "y"}]})
        assert "не нашёл «FVG»" in sent.messages[-1][0]
    finally:
        storage.FILE = original
        sent.restore()
