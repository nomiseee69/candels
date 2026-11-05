"""Короткий код для индикатора Pattern Finder: рисуется ровно то же, что файл .pine."""

import re

import pine
import scanner
from catalog import ALL_PATTERNS
from market import candles_to_df
from templates import template_shape
from test_eye import fake_market

SETTINGS = {"quality": "off", "fvg": "all", "ob_volume": False}


def all_patterns():
    out = []
    for name in ALL_PATTERNS:
        p = {"name": name, "builtin": True}
        if name in scanner.SHAPE_PATTERNS:
            p["shape"] = template_shape(name)
        out.append(p)
    return out


def markup_groups(seed, intervals=("1h", "4h", "1d")):
    candles = fake_market(seed, 1000)
    original = scanner.get_candles
    scanner.get_candles = lambda s, i, n: candles
    try:
        cards, _ = scanner.scan_user(all_patterns(), list(intervals), lambda s, i, n: candles, ["AUSDT"], SETTINGS)
        if not cards:
            return None
        coin = scanner.group_by_coin(cards)[0]
        return scanner.full_markup(all_patterns(), coin, list(intervals), SETTINGS)
    finally:
        scanner.get_candles = original


def same(a, b, key):
    if isinstance(a, float):
        return abs(a - b) <= 2e-5 * max(abs(a), abs(b), 1e-12) + 1e-12
    return a == b


def test_code_draws_exactly_what_the_file_draws():
    checked = 0
    for seed in range(8):
        groups = markup_groups(seed)
        if groups is None:
            continue
        _, _, tfs, objects, kept = pine._collect(groups, "1h")
        original = sorted((o for o in objects if id(o) in kept), key=lambda o: (-o[3], -o[4]))
        decoded = pine.decode_code("".join(pine.pine_code("AUSDT", "1h", groups)))
        assert decoded["tfs"] == tfs and decoded["symbol"] == "AUSDT"
        # объекты те же (по виду), с теми же координатами, цветами, стилями и текстами
        assert len(decoded["objects"]) == len(original)
        key = lambda d: (d.get("x", d.get("x1")), round(d.get("y", d.get("y1", d.get("top", 0))), 6), d.get("text", ""))
        want = sorted((dict(o[6].data, kind=o[5], tf=o[0], mode=o[1], cat=o[2]) for o in original), key=key)
        got = sorted(decoded["objects"], key=key)
        for w, g in zip(want, got):
            w = dict(w, text=pine._clean(w["text"]).replace("_", "") if "text" in w and w["text"] != "●" else w.get("text"))
            for k, v in w.items():
                if k == "text" and v is None:
                    continue
                assert same(v, g[k], k), (seed, k, v, g[k])
            checked += 1
        assert len(decoded["rows"]) == len(tfs)                       # панель-сводка: строка на ТФ
    assert checked > 50


def test_code_is_short_and_split_into_parts():
    groups = markup_groups(10)
    file_code = pine.pine_script("AUSDT", "1h", groups)
    parts = pine.pine_code("AUSDT", "1h", groups)
    assert sum(map(len, parts)) < len(file_code) / 3                    # в разы короче файла
    # много объектов — режется на части по ≤ PART_CHARS, склейка частей = исходный код
    text = "PF1,A,0,1,1,1h|" + "|".join(f"T,0,p,f,{pine.b36(i)},{pine.b36(i)},метка {i},y,l,t" for i in range(900))
    parts = pine._split_parts(text)
    assert len(parts) > 1 and all(len(p) <= pine.PART_CHARS for p in parts) and "".join(parts) == text


def test_too_much_history_is_trimmed_but_current_stays():
    groups = markup_groups(3)
    full = "".join(pine.pine_code("AUSDT", "1h", groups))
    records = full.split("|")
    current = sum(len(r) + 1 for r in records if r.split(",")[0] in ("PF1", "S") or r.split(",")[2] != "p")
    original = pine.PART_CHARS
    pine.PART_CHARS = (current + 300) // pine.MAX_PARTS                  # текущее влезает, вся история — нет
    try:
        assert len(full) > pine.PART_CHARS * pine.MAX_PARTS
        parts = pine.pine_code("AUSDT", "1h", groups)
        decoded = pine.decode_code("".join(parts))
        _, _, _, objects, kept = pine._collect(groups, "1h")
        now = sum(1 for o in objects if id(o) in kept and o[1] == "now")
        assert len(parts) <= pine.MAX_PARTS and all(len(p) <= pine.PART_CHARS for p in parts)
        assert sum(1 for o in decoded["objects"] if o["mode"] == "now") == now   # текущие — все на месте
        assert len(decoded["objects"]) < len(pine.decode_code(full)["objects"])  # убрана часть истории
    finally:
        pine.PART_CHARS = original


def test_indicator_understands_every_code_letter():
    """Индикатор в TradingView знает все буквы, которые пишет бот (цвета, стили, размеры, записи)."""
    source = open("tradingview/PatternFinder.pine", encoding="utf-8").read()
    assert source.startswith("//@version=6") and '"PF1"' in source
    for letter in pine.COLOR_LETTERS.values():
        assert f'"{letter}" => color.' in source or letter == "k"
    for name, code in pine.LABEL_STYLES.items():
        assert code == "l" or f'"{code}" => {name}' in source
    assert 's == "d" ? line.style_dashed : s == "t" ? line.style_dotted' in source
    assert 's == "r" ? extend.right' in source and 's == "t" ? size.tiny' in source
    for kind in ("S", "L", "B", "T", "D"):
        assert f'kind == "{kind}"' in source
    assert source.count("input.text_area") == pine.MAX_PARTS
    assert f"/ {pine.PRICE_SCALE}" in source and "* 1000)" in source
    for tf in pine.INTERVAL_SECONDS:                                    # переключатели ТФ — как в файле
        assert f'"{tf}" => {pine.INTERVAL_SECONDS[tf]}' in source or tf == "1w"
    # каждый блок с отступом — кратен 4 пробелам (Pine строг к отступам)
    for line in source.splitlines():
        indent = len(line) - len(line.lstrip(" "))
        assert indent % 4 == 0, line


def test_bot_sends_indicator_once_then_only_code(tmp_path):
    import bot
    import storage
    from test_menus import Recorder, use_storage
    rec = Recorder()
    original = use_storage(tmp_path)
    saved = bot.send_document, bot.full_markup
    docs = []
    bot.send_document = lambda path, caption, chat_id, filename=None, keyboard=None, reply_to=None: docs.append(filename)
    groups = markup_groups(10)
    bot.full_markup = lambda patterns, coin, intervals, settings: groups
    try:
        coin = {"symbol": "AUSDT", "frames": [{"interval": "1h", "df": groups[0][0], "items": [], "history": False}]}
        cid = bot.remember_cards(5, [coin])[0]
        bot.send_pine(5, cid)
        assert docs == ["PatternFinder.pine.txt"]                         # первый раз — сам индикатор
        code = [t for t, _ in rec.sent if "<code>" in t]
        assert code and re.search(r"<code>PF1,AUSDT,", code[-1])
        bot.send_pine(5, cid)
        assert docs == ["PatternFinder.pine.txt"]                         # второй раз — только код
        storage.toggle_setting(5, "markup")                               # «файл .pine» — как раньше
        bot.send_pine(5, cid)
        assert docs[-1] == "AUSDT_patterns.pine.txt"
    finally:
        bot.send_document, bot.full_markup = saved
        rec.restore()
        storage.FILE = original
