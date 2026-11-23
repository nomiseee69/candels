"""Replay extended_charts.rank and record each candidate before the photo gates."""
import argparse
import json
import os
import sys
import tempfile
from contextlib import ExitStack, chdir
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import extended_charts as exam


SAMPLES = {"bearish_fvg": ("FVG", "bearish_fvg"), "falling_wedge": ("Клин", "falling_wedge"),
           "dark_channel_down": ("Канал", "channel_down"), "dark_triangle": ("Треугольник", "symmetric")}


def heldout(cache):
    for path in sorted(cache.glob("*-USD-*.json")):
        tf = path.stem.rsplit("-", 1)[1]
        start = pd.Timestamp("2024-01-01" if tf == "1d" else "2025-03-01", tz="UTC")
        df = pd.DataFrame(json.loads(path.read_bytes()), columns=["time", "low", "high", "open", "close", "volume"])
        df["open_time"] = pd.to_datetime(df.time, unit="s", utc=True)
        df = df.set_index("open_time").sort_index().loc[start:].iloc[:240].reset_index()
        assert list(df.open_time) == list(pd.date_range(start, periods=240, freq="D" if tf == "1d" else "h"))
        yield path, df


def fvg_trace(df, reference, eligible):
    import fingerprint as fp
    import smc
    from structure import atr
    sizes = atr(df)
    market = {it["start"]: it for it in smc.find_all_fvg(df, lookback=len(df), include_filled=True)}
    rows = []
    for i in range(len(df) - 2):
        gap = exam.independent_gap(df, i)
        if not gap:
            continue
        item = market.get(i)
        fresh = i >= len(df) - smc.LOOKBACK - 2
        distance = max(gap[1] - df.close.iloc[-1], df.close.iloc[-1] - gap[2], 0) / df.close.iloc[-1]
        active = item and item["filled_at"] is None and (fresh or distance <= smc.NEAR_PRICE)
        if i not in eligible and not active:
            continue
        row = dict(start=i, kind=gap[0], labelled=i in eligible, gap_atr=(gap[2]-gap[1])/sizes[i+1],
                   active=bool(active), fresh=fresh, distance=distance)
        if item:
            features = fp.features_of("fvg", df, item)
            row.update(score=item["score"], features=features, numeric=fp.feature_similarity(reference["features"], features),
                       visual=fp.visual_similarity(reference, df, item), similarity=fp.similarity(reference, df, item))
        rows.append(row)
    return rows


def lines_trace(df, reference, name, kind):
    from scanner import MIN_SCORE
    from similarity import best_similarity
    from trendlines import FAMILY_BY_PATTERN, WINDOWS, find_line_pattern
    rows = []
    for window in WINDOWS:
        if window > len(df):
            continue
        # A one-window call exposes all active, fitted boundaries before the best-window reduction.
        item = find_line_pattern(df, FAMILY_BY_PATTERN[name], kind, windows=[window],
                                 similarity=lambda d, it: best_similarity(d, it, [reference]))
        if item:
            rows.append(dict(window=window, start=item["start"], kind=item["kind"], touches=item["touches"],
                             violations=item["violations"], score=item["score"], similarity=item["similarity"],
                             labelled=item["kind"] == kind and item["score"] >= MIN_SCORE))
    return rows


def stage_item(item):
    if item is None:
        return None
    return {k: item[k] for k in ("start", "kind", "window", "score", "similarity", "grade") if k in item}


def watch_detector(original, log):
    def traced(*args, **kwargs):
        item = original(*args, **kwargs)
        log.append(stage_item(item))
        return item
    return traced


def card_stages(cards):
    return {f"{c['symbol']}/{c['interval']}": [stage_item(i) for i in c["items"]] for c in cards}


def evaluate(engine, dataset, cache, registry):
    import scanner
    from similarity import reference_from_screenshot
    from fingerprint import screenshot_fingerprint
    from vision import extract_marks
    import bot
    loader = exam.markets if dataset == "original" else heldout
    feeds = list(loader(cache))
    assert len(feeds) == 6
    source_hashes = {p.name: exam.digest(p) for p, _ in feeds}
    if dataset == "original":
        assert all(any(r.get("file") == f and r.get("sha256") == h for r in registry) for f, h in source_hashes.items())
    data = {(p.stem.split("-")[0], p.stem.split("-")[-1]): df for p, df in feeds}
    records = []
    for sample, (name, kind) in SAMPLES.items():
        collected = []
        raw, gated, scanned, pre_group = [], [], [], {}
        original_scan, original_group = bot.scan_user, bot.group_by_coin
        def traced_scan(*args, **kwargs):
            cards, skipped = original_scan(*args, **kwargs)
            scanned.append(card_stages(cards))
            return cards, skipped
        def traced_group(cards):
            pre_group[len(scanned) - 1] = card_stages(cards)
            return original_group(cards)
        with ExitStack() as stack:
            stack.enter_context(patch.object(exam, "markets", loader))
            stack.enter_context(patch.object(exam, "save", side_effect=lambda _, r: collected.extend(r)))
            for fn, log in (("find_fvg" if name == "FVG" else "find_line_pattern", raw),
                            ("find_rule" if name == "FVG" else "find_lines", gated)):
                stack.enter_context(patch.object(scanner, fn, watch_detector(getattr(scanner, fn), log)))
            stack.enter_context(patch.object(bot, "scan_user", traced_scan))
            stack.enter_context(patch.object(bot, "group_by_coin", traced_group))
            exam.rank(cache, engine, sample + ".png")
        assert len(raw) == len(gated) == 18 and len(scanned) == len(collected) == 6
        if dataset == "original" and engine.name in ("d64914a", "8a9b3f5"):
            saved = next(r for r in registry if r.get("run") == f"rank-{engine.name}-{sample}")
            assert saved["snapshots"] == collected
        path = str(engine / "tests/samples" / (sample + ".png"))
        if (engine / "screenshot_query.py").exists():
            from screenshot_query import screenshot_choices
            example, = [ex for _, ex in screenshot_choices(path) if ex["name"] == name]
        elif name == "FVG":
            df, zones, lines = extract_marks(path)
            reference, _ = screenshot_fingerprint(name, df, bot.drawn_levels(path), zones, lines)
            example = {"fingerprint": reference}
        else:
            example = {"reference": reference_from_screenshot(path)}
        for j, snap in enumerate(collected):
            stop, tf = snap["stop"], snap["interval"]
            traces = {}
            for symbol in ("BTC", "ETH", "SOL"):
                df = data[symbol, tf].iloc[:stop]
                key = symbol + "/" + tf
                if name == "FVG":
                    traces[key] = fvg_trace(df, example["fingerprint"], snap["eligible"][key])
                else:
                    traces[key] = lines_trace(df, example["reference"], name, kind)
            stages = {}
            for k, key in enumerate(traces):
                stages[key] = dict(detector_winner=raw[3*j+k], after_score_similarity=gated[3*j+k],
                                   after_user_filters=scanned[j].get(key, []),
                                   before_grouping=pre_group.get(j, {}).get(key, []))
            records.append(dict(sample=sample, kind=kind, **snap, candidates=traces, stages=stages))
    return {"dataset": dataset, "version": engine.name, "sha256": source_hashes, "queries": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    engine, original, holdout, output = [p.resolve() for p in (args.engine, args.original, args.holdout, args.output)]
    registry = [json.loads(s) for s in Path(__file__).with_name("extended_charts_results.jsonl").read_text(
        encoding="utf8").splitlines()]
    sys.path.insert(0, str(engine))
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    with tempfile.TemporaryDirectory() as folder, chdir(folder):
        results = [evaluate(engine, name, cache, registry) for name, cache in
                   (("original", original), ("holdout", holdout))]
    output.write_text(json.dumps(results, ensure_ascii=False, default=lambda v: v.item(), allow_nan=False), encoding="utf8")
