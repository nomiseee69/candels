"""Frozen multi-family photo replay. Run prepare once using the baseline engine, then replay per revision."""
import argparse
import json
import os
import sys
import tempfile
from contextlib import chdir
from pathlib import Path

import numpy as np
import pandas as pd

import extended_charts as old


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               default=lambda v: v.item(), indent=1), encoding="utf8")


def feeds(original, fresh):
    data = {("dev", p.stem): (p, df) for p, df in old.markets(original)}
    for p in sorted(fresh.glob("*-USD-*.json")):
        df = pd.DataFrame(json.loads(p.read_bytes()), columns=["time","low","high","open","close","volume"])
        df["open_time"] = pd.to_datetime(df.time, unit="s", utc=True)
        hourly = p.stem.endswith("1h")
        first = pd.Timestamp("2025-11-01" if hourly else "2025-01-01", tz="UTC")
        df = df[df.open_time >= first].sort_values("time").iloc[:287].reset_index(drop=True)
        assert list(df.open_time) == list(pd.date_range(first, periods=287, freq="h" if hourly else "D"))
        data["holdout", p.stem] = p, df
    assert len(data) == 12
    return data


def occurrences(name, df):
    from catalog import SHAPE_PATTERNS
    from screen_exam import cases
    from shapes_check import validate
    from structure import atr
    if name not in SHAPE_PATTERNS:
        for it in cases(name, df, len(df)):
            yield it, "history-rule"
        return
    sign = -1 if name in ("Двойное дно", "Перевёрнутая ГиП") else 1
    peaks = 2 if name in SHAPE_PATTERNS[:2] else 3
    for lo, hi, swings in old.swing_examples(df, sign, peaks):
        yield {"start":lo,"end":hi-1,"swings":swings}, "independent-extrema"
    sizes = atr(df).to_numpy()
    for width in (40,60,80):
        for start in range(0, len(df)-width+1, 4):
            it = validate(name, df, start, width, sizes)
            if it and it["confirmed"]:
                yield {**it, "start":start, "end":start+width-1}, "confirmed-shape-rule"


def extent(it):
    return int(it["start"]), int(it["end"] if "end" in it else it["start"]+it["window"]-1)


def monotone(n):
    # Overlapping ranges, identical slope, no reversals, impulses, gaps or swing levels.
    y = np.linspace(100, 130, n)
    return pd.DataFrame(dict(open=y, high=y+1., low=y-1., close=y+.1, volume=np.ones(n)))


def prepare(cache, markets):
    from catalog import ALL_PATTERNS, SHAPE_PATTERNS
    import screen_exam
    from wild_exam import render
    examples = []
    for split in ("dev","holdout"):
        series = [(symbol,df) for (s,symbol),(_,df) in markets.items() if s == split]
        for i,name in enumerate(ALL_PATTERNS):
            count = 0
            for symbol,df in series[i%6:] + series[:i%6]:
                for it,basis in occurrences(name, df):
                    start,end = extent(it)
                    if 119 <= end < len(df)-8 and start >= 10 and end > start:
                        break
                else:
                    continue
                lo = start if name in SHAPE_PATTERNS else max(0,start-35)
                hi = end+1 if name in SHAPE_PATTERNS else end+8
                part = df.iloc[lo:hi]
                mark = screen_exam._segment(it,lo)
                key = f"{split}-{i:02}-{count}"
                options = {} if count==0 else {"theme":"light","up":"#2962ff","down":"#f23645"}
                common = dict(id=key,split=split,name=name,kind=it.get("kind"),direction=it.get("direction"),
                              source=symbol,lo=lo,hi=hi,start=start,end=end,basis=basis,
                              utc=df.open_time[end].isoformat(),mark=mark)
                for role in ("positive","negative","ambiguous"):
                    image = cache / f"{key}-{role}.png"
                    if role=="positive":
                        bars,annotation = part,mark
                    elif role=="ambiguous":
                        # Less than the required context, from the actual event.
                        bars,annotation = df.iloc[end-6:end+1],(None,None)
                    else:
                        bars,annotation = monotone(len(part)),(None,None)
                        # Preserve the drawing geometry on the deliberately pattern-free tape.
                        if mark[0]:
                            kind,args = mark
                            price=lambda v: 100+30*(v-part.low.min())/max(part.high.max()-part.low.min(),1e-8)
                            if kind=="lines":
                                annotation=(kind, [(x,price(y),xx,price(yy)) for x,y,xx,yy in args])
                            elif kind=="hline":
                                annotation=(kind,(*args[:2],price(args[2])))
                            else:
                                annotation=(kind,(*args[:-2],price(args[-2]),price(args[-1])))
                    render(bars,str(image),"",annotation,**options)
                    examples.append({**common,"role":role,"image":image.name,"sha256":old.digest(image)})
                count+=1
                if count==3:
                    break
    save(cache/"cases.json",dict(sources={f"{s}/{k}":old.digest(p) for (s,k),(p,_) in markets.items()},examples=examples))


def path_score(a,b):
    """Independent full candle contour, bounded 0..1. No scanner fingerprint or ranking weights."""
    def contour(df):
        z=df[["high","low","close"]].to_numpy()
        if len(z)<2 or z.max()==z.min():
            return None
        z=(z-z.min())/(z.max()-z.min())
        return np.array([np.interp(np.linspace(0,1,32),np.linspace(0,1,len(z)),v) for v in z.T])
    x,y=contour(a),contour(b)
    return None if x is None or y is None else round(float(1-np.mean(abs(x-y))),4)


def search(example, row, df, reference_bars=None, *, by_photo=True):
    from scanner import scan_user
    from similarity import SIMILAR_ENOUGH
    tf=row["source"].split("-")[-1]
    data=df.iloc[:row.get("cutoff",row["end"])+1]
    candles=data[["time","open","high","low","close","volume"]].assign(time=lambda d:d.time*1000).values.tolist()
    cards,_=scan_user([example],[tf],symbols=["TEST"],fetch=lambda *_: candles,settings={
        "quality":"off","learn":False,"fvg":"all","ob_volume":False,"similar_only":by_photo,
        "photo_kind":example.get("kind") if by_photo else None})
    if not cards:
        return dict(status="no detector card")
    item=cards[0]["items"][0]
    score=item.get("score") if example.get("shape") else item.get("similarity")
    if any(example.get(k) and example[k]!=item.get(k) for k in ("kind","direction")):
        return dict(status="wrong direction/subtype")
    if by_photo and (score or 0)<SIMILAR_ENOUGH:
        return dict(status="photo similarity")
    a,b=extent(item)
    # Tolerate uncertainty of four bars in marking the starting candle (same as screen_exam).
    overlap=max(0,min(b,row["end"])-max(a,row["start"])+1)/max(1,max(b,row["end"])-min(a,row["start"])+1)
    located=abs(a-row["start"])<=4 if row["end"]-row["start"]<10 else overlap>=.5
    return dict(status="located" if located else "other location",start=a,end=b,score=round(float(score or item["score"]),4),
                contour=None if reference_bars is None else path_score(reference_bars,data.iloc[a:b+1]))


def replay(cache, markets, output):
    from catalog import SHAPE_PATTERNS
    from PIL import Image
    from templates import template_shape
    from screenshot_query import screenshot_choices
    from vision import UnreliableScreenshot, extract_marks
    manifest=json.loads((cache/"cases.json").read_text("utf8"))
    assert manifest["sources"]=={f"{s}/{k}":old.digest(p) for (s,k),(p,_) in markets.items()}
    refs={}
    results=[]
    for row in manifest["examples"]:
        path=cache/row["image"]
        assert old.digest(path)==row["sha256"]
        rec={k:row[k] for k in ("id","name","source","utc","basis","kind","direction","start","end","image","sha256","role","split")}
        try:
            choices=[ex for _,ex in screenshot_choices(str(path))]
        except UnreliableScreenshot as exc:
            choices=[]
            rec["refusal"]=str(exc)
        rec["choices"]=[(ex["name"],ex.get("kind"),ex.get("direction")) for ex in choices]
        target=[ex for ex in choices if ex["name"]==row["name"] and all(not row[k] or ex.get(k)==row[k] for k in ("kind","direction"))]
        rec["recognized"]=bool(target)
        if row["role"]=="positive":
            df=markets[row["split"],row["source"]][1]
            # Read pixel OHLC from the screenshot. Invert the known renderer axis
            # solely for scoring the annotated region, never for detection.
            bars=None
            try:
                pixels,_,_=extract_marks(str(path),include_x=True)
                if pixels is not None:
                    with Image.open(path) as image:
                        positions=((pixels.x/image.width-.01)/.91*(row["hi"]-row["lo"]+7)-1).round()+row["lo"]
                    bars=pixels[positions.between(row["start"],row["end"])]
            except UnreliableScreenshot:
                pass
            named={"name":row["name"],"kind":row["kind"],"direction":row["direction"]}
            if row["name"] in SHAPE_PATTERNS:
                named["shape"]=template_shape(row["name"])
            rec["market"]=search(named,row,df,by_photo=False)
            key=(row["name"],row["kind"],row["direction"])
            if target:
                rec["self"]=search(target[0],row,df,bars)
                if row["split"]=="dev":
                    refs.setdefault(key,(target[0],bars))
            if row["split"]=="holdout" and key in refs:
                ex,reference=refs[key]
                rec["transfer"]=search(ex,row,df,reference)
        results.append(rec)
    save(output,results)


def real_controls(cache, markets, output):
    """Wrong mark on real prices, or an interval without two extrema for an unmarked shape."""
    from catalog import ALL_PATTERNS, SHAPE_PATTERNS
    from screenshot_query import screenshot_choices
    from vision import UnreliableScreenshot
    from wild_exam import render
    from scipy.signal import find_peaks
    results = []
    for split in ("dev","holdout"):
        series = [(k,df) for (s,k),(_,df) in markets.items() if s == split]
        for index,name in enumerate(ALL_PATTERNS):
            symbol,df=series[index % len(series)]
            start=65
            mark=(None,None)
            reason="annotation above all candle highs"
            if name in SHAPE_PATTERNS:
                # No pair of distinct highs (M/H&S) or lows (W/inverse H&S), even at 1-bar resolution.
                high_side = name in ("Двойная вершина","Голова и плечи")
                series_price = df.high.to_numpy() if high_side else -df.low.to_numpy()
                positions=[a for a in range(len(df)-15) if len(find_peaks(series_price[a:a+15])[0])<2]
                if not positions:
                    # The 3-pivot H&S families additionally require a central head.
                    if name not in SHAPE_PATTERNS[2:]:
                        continue
                    positions=[a for a in range(len(df)-15) if len(find_peaks(series_price[a:a+15])[0])<3]
                if not positions:
                    continue
                start=positions[0]
                part=df.iloc[start:start+15]
                reason="too few distinct swing extrema inside image"
            else:
                part=df.iloc[start:start+48]
                span=part.high.max()-part.low.min()
                top=part.high.max()
                # Draw inside the renderer's 5% upper margin, entirely clear of the OHLC.
                low,high=top+.008*span,top+.043*span
                if name in ("BOS","CHoCH","EQH / EQL"):
                    mark=("hline",(8,40,high))
                elif name in ("FVG","Order Block","Breaker Block","Mitigation Block"):
                    mark=("zone",(12,low,high))
                elif name=="Range":
                    mark=("box",(8,40,low,high))
                else:
                    tilt=(high-low)/3
                    end_top=high-tilt if name!="Канал" else high
                    end_low=low+(tilt if name in ("Треугольник","Вымпел") else -tilt/2 if name=="Клин" else 0)
                    mark=("lines",((8,high,40,end_top),(8,low,40,end_low)))
            img=cache/f"real-negative-{split}-{index}.png"
            if not img.exists():
                render(part,str(img),"",mark)
            try:
                choices=[ex["name"] for _,ex in screenshot_choices(str(img))]
            except UnreliableScreenshot:
                choices=[]
            results.append(dict(split=split,name=name,source=symbol,start=start,utc=part.open_time.iloc[0].isoformat(),
                                image=img.name,sha256=old.digest(img),reason=reason,choices=choices))
    save(output,results)


def confirm_levels(cache, markets, output):
    """Observe the first possible live confirmation of the last equal-high/low pivot."""
    from structure import LIQUIDITY_WINDOW
    from screenshot_query import screenshot_choices
    from vision import UnreliableScreenshot
    rows = []
    for row in json.loads((cache/"cases.json").read_bytes())["examples"]:
        if row["name"]!="EQH / EQL" or row["role"]!="positive":
            continue
        try:
            target=next(ex for _,ex in screenshot_choices(str(cache/row["image"])) if ex.get("kind")==row["kind"])
        except (UnreliableScreenshot,StopIteration):
            continue
        df=markets[row["split"],row["source"]][1]
        check={**row,"cutoff":row["end"]+LIQUIDITY_WINDOW}
        rows.append(dict(id=row["id"],confirmed_at=check["cutoff"],photo=search(target,check,df),
                         market=search({"name":row["name"],"kind":row["kind"]},check,df,by_photo=False)))
    save(output,rows)


def legacy_transfer(cache, original, engine, markets, output):
    """Additional fixed photo bank, selected without seeing prospective target scores."""
    from screenshot_query import screenshot_choices
    from vision import UnreliableScreenshot
    bank = {}
    for row in json.loads((original/"manifest.json").read_bytes()):
        if row["group"] not in ("known","review","rule","synthetic","gap"):
            continue
        if row.get("style") not in (None,"dark"):
            continue
        src = engine/"tests/samples" if row["source"]=="repo" else original/"images"
        path = src/row["file"]
        assert old.digest(path)==row["sha256"]
        try:
            choices=screenshot_choices(str(path))
        except UnreliableScreenshot:
            continue
        for _,ex in choices:
            if ex["name"]==row["name"] and (not row.get("kind") or ex.get("kind")==row["kind"]):
                bank.setdefault((ex["name"],ex.get("kind"),ex.get("direction")),(ex,row["file"]))
    rows=[]
    for row in json.loads((cache/"cases.json").read_bytes())["examples"]:
        if row["role"]!="positive" or row["split"]!="holdout":
            continue
        key=(row["name"],row["kind"],row["direction"])
        if key not in bank:
            continue
        ex,source=bank[key]
        df=markets["holdout",row["source"]][1]
        clean=monotone(len(df)).assign(time=df.time,open_time=df.open_time)
        rows.append(dict(id=row["id"],reference=source,photo=search(ex,row,df),negative=search(ex,row,clean)))
    save(output,rows)


def diagnose(cache, markets, output):
    """Inspect the unthresholded photo evidence at the labelled site and the live rule winner."""
    from catalog import SHAPE_PATTERNS, LINE_PATTERNS
    from fingerprint import all_items, marked_items, similarity
    from matcher import find_best_match
    from scanner import WINDOWS, RECENT, build_detectors
    from screenshot_query import screenshot_choices
    from shapes_check import landmark_closes, validate
    from structure import atr
    from vision import UnreliableScreenshot, extract_marks
    bank, report = {}, []
    for row in json.loads((cache/"cases.json").read_bytes())["examples"]:
        if row["role"]!="positive":
            continue
        key=(row["name"],row["kind"],row["direction"])
        path=str(cache/row["image"])
        try:
            choices=screenshot_choices(path)
        except UnreliableScreenshot:
            choices=[]
        ex=next((ex for _,ex in choices if (ex["name"],ex.get("kind"),ex.get("direction"))==key),None)
        if row["split"]=="dev":
            if ex:
                bank.setdefault(key,ex)
            continue
        df=markets["holdout",row["source"]][1].iloc[:row["end"]+1]
        rec=dict(id=row["id"])
        if ex is None:
            try:
                photo,zones,lines=extract_marks(path)
                rec["marks"]={"candles":0 if photo is None else len(photo),"zones":len(zones),"lines":len(lines)}
                if photo is not None and key[0] not in SHAPE_PATTERNS+LINE_PATTERNS:
                    rec["mark_evidence"]=[(round(s,4),it.get("kind"),it.get("direction"))
                                          for s,it in marked_items(key[0],photo,zones,lines)]
            except UnreliableScreenshot as exc:
                rec["refusal"]=str(exc)
        for mode,ref in (("self",ex),("transfer",bank.get(key))):
            if ref is None:
                continue
            jobs,_=build_detectors([ref],{"fvg":"all","quality":"off","ob_volume":False,"similar_only":False})
            raw=jobs[0][1](df)
            evidence={"live_before_photo":None if raw is None else {
                k:raw[k] for k in ("start","kind","direction","score","similarity") if k in raw}}
            if key[0] in SHAPE_PATTERNS:
                scores=[]
                sizes=atr(df).to_numpy()
                for width in set(WINDOWS)|{ref["shape_window"]}:
                    for end in range(len(df),max(width-1,len(df)-RECENT),-1):
                        figure=validate(key[0],df,end-width,width,sizes)
                        if figure:
                            closes=landmark_closes(df,figure).tolist()
                            if len(closes)>=8:
                                scores.append(find_best_match(closes,ref["shape"],len(closes))[1])
                evidence["best_valid_correlation"]=max(scores,default=None)
            elif ref.get("fingerprint"):
                candidates=[it for it in all_items(key[0],df) if it.get("kind")==key[1]
                            and (not key[2] or it.get("direction")==key[2]) and abs(it["start"]-row["start"])<=4]
                evidence["at_site"]=[{"start":it["start"],"score":it["score"],
                                      "similarity":similarity(ref["fingerprint"],df,it),
                                      **{k:it[k] for k in ("strong","swept_at","broken_at","filled_at") if k in it}}
                                     for it in candidates]
            rec[mode]=evidence
        report.append(rec)
    save(output,report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("action","engine","original","fresh","cache","output"):
        parser.add_argument(key)
    args=parser.parse_args()
    engine,orig,fresh,cache,out=[Path(v).resolve() for v in (args.engine,args.original,args.fresh,args.cache,args.output)]
    sys.path.insert(0,str(engine))
    os.environ["PYTHON_DOTENV_DISABLED"]="1"
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as folder,chdir(folder):
        market=feeds(orig,fresh)
        if args.action=="prepare":
            prepare(cache,market)
        elif args.action=="replay":
            replay(cache,market,out)
        elif args.action=="controls":
            real_controls(cache,market,out)
        elif args.action=="levels":
            confirm_levels(cache,market,out)
        elif args.action=="bank":
            legacy_transfer(cache,orig,engine,market,out)
        elif args.action=="diagnose":
            diagnose(cache,market,out)
        else:
            parser.error("unknown evaluation action")
