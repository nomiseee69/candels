"""Offline paired evaluation. See EXTENDED_CHARTS.md for inputs and invocation."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markets(cache):
    for path in sorted(cache.glob("*-USD-*.json")):
        df = pd.DataFrame(json.loads(path.read_bytes()), columns=["time", "low", "high", "open", "close", "volume"])
        df["open_time"] = pd.to_datetime(df.time, unit="s", utc=True)
        tf = path.stem.split("-")[-1]
        start = pd.Timestamp("2023-01-01" if tf == "1d" else "2024-06-01", tz="UTC")
        df = df.set_index("open_time").sort_index().loc[start:].iloc[:240].reset_index()
        assert list(df.open_time) == list(pd.date_range(start, periods=240, freq="D" if tf == "1d" else "h"))
        yield path, df


def independent_gap(df, i):
    if df.low[i + 2] > df.high[i]:
        return "bullish_fvg", df.high[i], df.low[i + 2]
    if df.high[i + 2] < df.low[i]:
        return "bearish_fvg", df.high[i + 2], df.low[i]
    return None


def swing_examples(df, sign, count):
    """Independent candidate selector: alternating extrema, matched shoulders, neckline break."""
    import numpy as np
    from scipy.signal import find_peaks
    y = sign * df.close.to_numpy()
    smooth = pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy()
    scale = float((df.high - df.low).median())
    peaks, _ = find_peaks(smooth, distance=4, prominence=0.8*scale)
    for j in range(len(peaks) - count + 1):
        tops = peaks[j:j + count]
        first, last = int(tops[0]), int(tops[-1])
        if first < 12 or last + 12 >= len(y) or last - first < 12:
            continue
        dips = [int(a + np.argmin(y[a:b+1])) for a, b in zip(tops, tops[1:])]
        depth = min(y[first], y[last]) - max(y[dips])
        if depth < scale or abs(y[first] - y[last]) > .45*depth:
            continue
        if min(y[first-12:first]) > y[first] - depth or min(y[last+1:last+13]) >= min(y[dips]):
            continue
        if count == 3 and (y[tops[1]] - max(y[first], y[last]) < .4*depth or
                           abs(y[dips[0]] - y[dips[1]]) > .6*depth):
            continue
        yield first-12, last+13, sorted(list(map(int, tops)) + dips)


def prepare(cache, samples):
    import screen_exam as exam
    import wild_exam
    from catalog import ALL_PATTERNS, SHAPE_PATTERNS
    out = cache / "images"
    out.mkdir(exist_ok=True)
    rows = []
    def add(path, source, group, name, kind=None, **meta):
        rows.append(dict(file=path.name, sha256=digest(path), source=source, group=group,
                         name=name, kind=kind, **meta))

    labels = {"falling_wedge": ("Клин", "falling_wedge"), "bearish_fvg": ("FVG", "bearish_fvg"),
              "dark_triangle": ("Треугольник", "symmetric"), "dark_channel_down": ("Канал", "channel_down"),
              "dark_avax_lines": ("Двойное дно", None), "user_fvg_box_low": ("FVG", "bearish_fvg"),
              "user_fvg_box_side": ("FVG", "bearish_fvg")}
    for path in sorted(samples.glob("*.png")):
        group = "known" if path.stem in labels else "ambiguous"
        if path.stem == "cluttered_structure":
            group = "reject"
        add(path, "repo", group, *labels.get(path.stem, (None, None)))

    styles = {"dark": {}, "light-blue": {"theme": "light", "up": "#2962ff", "down": "#f23645"},
              "phone": {"size": (4.6, 8.2), "volume": True}, "mono": {"theme": "light", "up": "#ffffff",
              "down": "#000000", "edge": "#000000"}}
    def draw(source, group, name, kind, df, drawing, style, **meta):
        path = out / f"frame-{len(rows):03d}.png"
        wild_exam.render(df, str(path), "", drawing, **styles[style])
        add(path, source, group, name, kind, style=style, candles=len(df), **meta)

    series = list(markets(cache))
    assert len(series) == 6
    for path, df in series:
        # One wide gap of each polarity, then a near miss in an overlapping triplet.
        for kind in ("bearish_fvg", "bullish_fvg", "overlap"):
            candidates = []
            for i in range(30, 93):
                gap = independent_gap(df, i)
                if gap and gap[0] == kind:
                    candidates.append((gap[2] - gap[1], i, gap[1], gap[2]))
                elif gap is None and kind == "overlap":
                    low, high = max(df.low[i], df.low[i+2]), min(df.high[i], df.high[i+2])
                    candidates.append((high - low, i, low, high))
            if candidates:
                _, i, low, high = max(candidates)
                for style in styles:
                    draw(path.stem, "near_miss" if kind == "overlap" else "gap", "FVG", kind,
                         df.iloc[i-24:i+12], ("zone", (24, low, high)), style, first=i,
                         utc=df.open_time[i].isoformat(), bounds=[low, high])

    for index, name in enumerate(ALL_PATTERNS):
        # Silver labels are proposed on OHLC by an existing detector. Never count as gold.
        for path, df in series[index % 6:] + series[:index % 6]:
            items = exam.cases(name, df, 1)
            if items:
                item = items[0]
                first = int(item["start"])
                end = int(item.get("end", first + item.get("window", 1) - 1))
                lo = max(0, first - 35)
                window = df.iloc[lo:end+11]
                for style in ("dark", "light-blue"):
                    draw(path.stem, "rule", name, item.get("kind"), window, exam._segment(item, lo), style,
                         first=first, utc=df.open_time[first].isoformat(), direction=item.get("direction"))
                break
        if name in SHAPE_PATTERNS:
            df, key = exam.planted(name, 373 + index)
            for style in ("dark", "mono"):
                draw(f"planted-{373 + index}", "synthetic", name, None, df, (None, None), style, key=key)
    for name, sign, count in (("Двойное дно", -1, 2), ("Двойная вершина", 1, 2),
                              ("Голова и плечи", 1, 3), ("Перевёрнутая ГиП", -1, 3)):
        for path, df in series:
            match = next(swing_examples(df, sign, count), None)
            if match:
                lo, hi, swings = match
                for style in ("dark", "light-blue"):
                    draw(path.stem, "swing", name, None, df.iloc[lo:hi], (None,None), style,
                         first=lo, utc=df.open_time[lo].isoformat(), swings=swings)
                break
    history = pd.DataFrame(json.loads((cache / "BTC-2021.json").read_bytes()),
                           columns=["time","low","high","open","close","volume"])
    history.index = pd.to_datetime(history.time, unit="s", utc=True)
    history = history.sort_index()
    for name, first, last in (("Голова и плечи","2021-03-01","2021-05-23"),
                              ("Двойное дно","2021-06-13","2021-08-07")):
        window = history.loc[first:last]
        assert len(window) == len(pd.date_range(first,last))
        for style in ("dark","light-blue"):
            draw("BTC-2021", "review", name, None, window, (None,None), style, utc=first, end=last)
    add(cache / "rink-fig4.png", "Rink2025-fig4", "reject_line", "Голова и плечи",
        basis="Три пика, средний выше; пробой шеи. На рисунке только линии, без OHLC.")
    save(cache / "manifest.json", rows)


def recognize(cache, engine):
    from PIL import Image
    from fingerprint import make_fingerprint
    from screenshot_query import screenshot_choices
    from vision import PANEL, UnreliableScreenshot, extract_marks
    results = []
    for row in json.loads((cache / "manifest.json").read_text(encoding="utf8")):
        folder = engine / "tests/samples" if row["source"] == "repo" else cache / "images"
        if row["group"] == "reject_line":
            folder = cache
        path = folder / row["file"]
        assert digest(path) == row["sha256"]
        entry = {"file": row["file"]}
        try:
            df, zones, lines = extract_marks(str(path), include_x=True)
            entry.update(candles=0 if df is None else len(df), zones=len(zones), lines=len(lines))
            candles = df.drop(columns="x") if df is not None else None
            def capture(name, candles, item):
                if row["group"] == "gap" and name == "FVG" and not PANEL.cropped:
                    # Invert the renderer's x-axis. All tagged triplets start at bar 24.
                    with Image.open(path) as img:
                        x_fraction = df.x.iloc[int(item["start"])] / img.width
                    observed_bar = (x_fraction - .01) / .91 * (row["candles"] + 7) - 1
                    entry.setdefault("start_error_bars", []).append(round(float(observed_bar - 24), 2))
                return make_fingerprint(name, candles, item)
            with patch("screenshot_query.extract_marks", return_value=(candles, zones, lines)), patch(
                    "screenshot_query.make_fingerprint", side_effect=capture):
                choices = screenshot_choices(str(path))
            entry["choices"] = [[ex["name"], ex.get("kind"), ex.get("direction")] for _, ex in choices]
        except UnreliableScreenshot as error:
            entry.update(choices=[], refusal=str(error))
        results.append(entry)
    save(cache / "recognition.json", results)


def rank(cache, engine, sample="bearish_fvg.png", strict=False):
    import bot
    import scanner
    from fingerprint import screenshot_fingerprint
    from storage import DEFAULT_SETTINGS
    from vision import extract_marks
    from similarity import reference_from_screenshot
    from trendlines import kind_from_drawing
    from vision import analyze_screenshot
    img = str(engine / "tests/samples" / sample)
    name = {"bearish_fvg.png":"FVG", "falling_wedge.png":"Клин", "dark_triangle.png":"Треугольник",
            "dark_channel_down.png":"Канал"}[sample]
    if (engine / "screenshot_query.py").exists():
        from screenshot_query import screenshot_choices
        matches = [ex for _, ex in screenshot_choices(img) if ex["name"] == name]
        example, = matches
    elif name == "FVG":
        df, zones, lines = extract_marks(img)
        fp, item = screenshot_fingerprint("FVG", df, bot.drawn_levels(img), zones, lines)
        example = {"name": "FVG", "kind": item["kind"], "fingerprint": fp}
    else:
        example = {"name":name, "kind":kind_from_drawing(analyze_screenshot(img).lines),
                   "reference":reference_from_screenshot(img)}
    feeds = {(p.stem.split("-")[0], p.stem.split("-")[-1]): df for p, df in markets(cache)}
    result = []
    for stop, interval in product((160, 200, 240), ("1h", "1d")):
        def fetch(symbol, tf, limit):
            return feeds[symbol, tf].iloc[:stop].tail(limit)[["time","open","high","low","close","volume"]].assign(
                time=lambda d: d.time * 1000).values.tolist()

        output = []
        settings = {**DEFAULT_SETTINGS, "fvg": "all", "quality": "off", "learn": False,
                    "similar_only": strict}
        with patch.multiple(bot, get_patterns=lambda _: [example], get_settings=lambda _: settings,
                            get_intervals=lambda _: [interval], get_coins=lambda _: 3,
                            top_symbols=lambda _: ["BTC","ETH","SOL"], user_judge=lambda *_: None,
                            search_patterns=lambda *args: [example], send_message=lambda *args: 1,
                            edit_message=lambda *args: True, send_cards=lambda _, coins, *args: output.extend(coins)
                            ), patch.object(bot.billing, "use_scan", return_value=True), patch.object(
                            bot.billing, "allowed_patterns", side_effect=lambda _, ps: ps), patch.object(
                            scanner, "get_candles", fetch):
            if (engine / "screenshot_query.py").exists():
                bot.run_scan(1, example)
            else:
                bot.run_scan(1)
        # Independent minimal relevance criterion, not a human judgement of overall appearance.
        eligible = {}
        for (symbol, tf), df in feeds.items():
            if tf != interval:
                continue
            starts = []
            for i in range(stop - 2) if name == "FVG" else ():
                gap = independent_gap(df, i)
                if gap and gap[0] == "bearish_fvg":
                    low, high = gap[1:]
                    mid = (low + high)/2
                    close = df.close[stop-1]
                    near = max(low-close, close-high, 0)/close <= .01
                    if (i >= stop-32 or near) and not (df.high.iloc[i+3:stop] >= mid).any():
                        starts.append(i)
            eligible[f"{symbol}/{tf}"] = starts
        cards = []
        for coin in output:
            for frame in coin["frames"]:
                for item in frame["items"]:
                    key = coin["symbol"] + "/" + frame["interval"]
                    relevant = item["start"] in eligible[key] and item["kind"] == "bearish_fvg" if name == "FVG" else None
                    cards.append(dict(market=key, start=int(item["start"]), kind=item["kind"],
                                      similarity=round(item["similarity"], 4), relevant=relevant,
                                      subtype_match=item["kind"] == example["kind"]))
        result.append(dict(stop=stop, interval=interval, eligible=eligible, cards=cards))
    variant = "-strict" if strict else ""
    save(cache / f"rank-{engine.name}{variant}-{Path(sample).stem}.json", result)


def export(cache, output):
    """Сохранить источники, метки и наблюдения. Сырые котировки и изображения остаются в кэше."""
    def read(name):
        return json.loads((cache / name).read_text(encoding="utf8"))
    records = []
    for path, _ in markets(cache):
        symbol, tf = path.stem.rsplit("-", 1)
        start, end, step = ("2023-01-01", "2023-08-29", 86400) if tf == "1d" else ("2024-06-01", "2024-06-11", 3600)
        url = (f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={step}"
               f"&start={start}T00:00:00Z&end={end}T00:00:00Z")
        records.append(dict(record="source", id=path.stem, file=path.name, sha256=digest(path), url=url))
    history = cache / "BTC-2021.json"
    records.append(dict(record="source", id="BTC-2021", file=history.name, sha256=digest(history),
                        url="https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400"
                            "&start=2021-03-01T00:00:00Z&end=2021-08-15T00:00:00Z"))
    records.append(dict(record="source", id="Rink2025-fig4", file="rink-fig4.png",
                        url="https://media.springernature.com/full/springer-static/image/art%3A10.1186%2F"
                            "s40854-025-00763-2/MediaObjects/40854_2025_763_Fig4_HTML.png",
                        doi="10.1186/s40854-025-00763-2", author="K. Rink", license="CC BY 4.0",
                        sha256=digest(cache / "rink-fig4.png")))
    for label, actual in zip(read("manifest.json"), read("recognition.json"), strict=True):
        assert label["file"] == actual["file"]
        record = dict(record="image", split="development" if label["source"] == "repo" else "evaluation",
                      **label, observed=actual)
        if label["group"] == "swing" and label["name"] == "Двойное дно":
            record["review"] = "Неоднозначно: резкое снижение и боковик; два закрытия не задают явную W."
        records.append(record)
    for version in ("d64914a", "8a9b3f5"):
        records.append(dict(record="guided", version=version, results=read(f"guided-{version}.json")))
    for filename in sorted(cache.glob("rank-*-*.json")):
        records.append(dict(record="ranking", run=filename.stem, snapshots=read(filename.name)))
    with output.open("w", encoding="utf8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "recognize", "rank", "guided", "export"])
    parser.add_argument("cache", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--output", type=Path, default=Path("extended_charts_results.jsonl"))
    args = parser.parse_args()
    cache, engine, output = args.cache.resolve(), args.engine.resolve(), args.output.resolve()
    sys.path.insert(0, str(engine))
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    with tempfile.TemporaryDirectory() as folder:
        original = os.getcwd()
        os.chdir(folder)
        try:
            if args.action == "prepare":
                prepare(cache, engine / "tests/samples")
            elif args.action == "rank":
                for sample in ("bearish_fvg.png","falling_wedge.png","dark_triangle.png","dark_channel_down.png"):
                    rank(cache, engine, sample)
                    if not (engine / "screenshot_query.py").exists():
                        rank(cache, engine, sample, strict=True)
            elif args.action == "guided":
                import screen_exam
                save(cache / f"guided-{engine.name}.json", screen_exam.run(per_pattern=2, seeds=(1, 2)))
            elif args.action == "export":
                export(cache, output)
            else:
                recognize(cache, engine)
        finally:
            os.chdir(original)
