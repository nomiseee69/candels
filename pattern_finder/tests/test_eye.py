"""«Глаз» бота: HOG или нейросеть (ONNX). Настоящую нейросеть тесты не скачивают —
вместо неё подделка с тем же интерфейсом, что у onnxruntime."""

import json

import numpy as np
import pandas as pd

import eye_check
import visual
from test_examples import ob_df


class FakeSession:
    """Как onnxruntime.InferenceSession: get_inputs()[0].name и run(None, {вход: картинка}).
    «Описание» — уменьшенная картинка (похожие картинки → похожие числа), форма как у DINOv2."""

    def __init__(self, ndim=3):
        self.ndim = ndim
        self.shapes = []

    def get_inputs(self):
        return [type("Input", (), {"name": "pixel_values"})()]

    def run(self, outputs, feed):
        batch = feed["pixel_values"]
        self.shapes.append((batch.shape, batch.dtype))
        small = batch[0].reshape(3, 28, 8, 28, 8).mean(axis=(2, 4)).ravel()     # 3×28×28 признаков
        if self.ndim == 3:
            return [np.stack([small, small * 0])[None]]                         # [1, токены, признаки]
        if self.ndim == 4:
            return [small.reshape(1, 3, 28, 28)]
        return [small[None]]


def candles(seed=1, n=60):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.roll(c, 1)
    o[0] = c[0]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5, "close": c})


def test_neural_eye_input_and_outputs():
    for ndim in (2, 3, 4):
        session = FakeSession(ndim)
        eye = visual.NeuralEye(session)
        vector = eye(visual.normalize_ohlc(candles()))
        assert session.shapes[-1] == ((1, 3, 224, 224), np.float32)      # как ждёт сеть
        assert abs(np.linalg.norm(vector) - 1) < 1e-6


def test_switching_eyes_and_cache():
    ohlc = visual.normalize_ohlc(candles())
    try:
        hog = visual.look(ohlc)
        visual.use_eye("нейросеть (тест)", visual.NeuralEye(FakeSession()), 0.6)
        neural = visual.look(ohlc)
        assert len(neural) != len(hog) and visual.EYE["random_cos"] == 0.6
        assert visual.similarity(neural, neural) == 1.0
        assert visual.similarity(neural, visual.look(visual.normalize_ohlc(candles(seed=7)))) < 1.0
    finally:
        visual.use_hog()
    assert visual.EYE["name"] == "HOG" and len(visual.look(ohlc)) == len(hog)


def test_render_color_looks_like_tradingview():
    img = visual.render_color(visual.normalize_ohlc(candles()))
    assert img.shape == (224, 224, 3)
    assert tuple(img[0, 0]) == (19, 23, 34)                               # тёмный фон
    colors = {tuple(p) for p in img.reshape(-1, 3)}
    assert (8, 153, 129) in colors and (242, 54, 69) in colors            # зелёные и красные свечи


def test_load_config_falls_back_to_hog(tmp_path):
    path = tmp_path / "ai.json"
    path.write_text(json.dumps({"eye": "neural", "model": str(tmp_path / "нет.onnx"), "random_cos": 0.5}))
    assert visual.load_config(str(path)) == "HOG"                         # модели нет — работаем дальше
    assert visual.load_config(str(tmp_path / "missing.json")) == "HOG"


def test_exam_counts_same_pattern_neighbours():
    """Глаз, который различает формы, получает точность выше угадывания."""
    rng = np.random.default_rng(0)
    waves = [("волна", np.sin(np.linspace(0, 6, 60) + rng.normal(0, 0.1))) for _ in range(12)]
    ramps = [("рост", np.linspace(0, 1, 60) + rng.normal(0, 0.03, 60)) for _ in range(12)]
    windows = []
    for name, closes in waves + ramps:
        df = pd.DataFrame({"open": closes - 0.02, "high": closes + 0.05, "low": closes - 0.05, "close": closes})
        windows.append((name, visual.normalize_ohlc(df)))
    result = eye_check.evaluate(windows, visual.hog_describe)
    assert result["precision"] > 0.9 and result["chance"] < 0.5 and result["cases"] == 24
    assert result["same_cos"] > result["random_cos"]
    worse = {**result, "precision": result["precision"] - 0.1}
    assert eye_check.decide(worse, result) and not eye_check.decide(result, result)


def test_exam_collects_real_pattern_cases():
    t0 = 1_780_000_000_000

    def raw(df):
        return [[t0 + i * 14_400_000, o, h, lo, c, v] for i, (o, h, lo, c, v) in
                enumerate(zip(df["open"], df["high"], df["low"], df["close"], df["volume"]))]

    data = {f"S{i}": raw(pd.concat([ob_df(8, 3, seed=i)] * 3, ignore_index=True)) for i in range(4)}
    windows = eye_check.collect(data, patterns=["Order Block", "BOS"])
    names = {name for name, _ in windows}
    assert names and names <= {"Order Block", "BOS"}
    assert all(o.min() >= 0 and o.max() <= 1 for _, o in windows)


def test_fingerprint_uses_current_eye():
    import fingerprint as fpm
    from blocks import all_blocks
    example = ob_df(8, 3, seed=1)
    ref = [b for b in all_blocks(example) if b["kind"] == "bullish_ob"][-1]
    fp = fpm.make_fingerprint("Order Block", example, {**ref, "name": "Order Block"})
    market = ob_df(8, 3, seed=5)
    block = [b for b in all_blocks(market) if b["kind"] == "bullish_ob"][-1]
    try:
        visual.use_eye("нейросеть (тест)", visual.NeuralEye(FakeSession()), 0.5)
        neural = fpm.visual_similarity(fp, market, block)
    finally:
        visual.use_hog()
    assert neural is not None and 0 <= neural <= 1
    assert fpm.visual_similarity(fp, market, block) is not None


def test_setup_ai_turns_neural_on_only_if_better(tmp_path):
    import os
    import setup_ai
    saved = (setup_ai.install_onnxruntime, setup_ai.get_model, setup_ai.exam, visual.load_neural)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    eye = visual.NeuralEye(FakeSession())
    setup_ai.install_onnxruntime = lambda: True
    setup_ai.get_model = lambda: ("DINOv2-small", "models/x.onnx", eye)
    hog = {"precision": 0.30, "random_cos": 0.41}
    try:
        setup_ai.exam = lambda e: (hog, {"precision": 0.45, "random_cos": 0.62})
        setup_ai.main(["--dino"])
        config = json.load(open("ai.json", encoding="utf-8"))
        assert config["eye"] == "neural" and config["random_cos"] == 0.62
        visual.load_neural = lambda path: eye                  # бот при запуске включает нейросеть
        assert visual.load_config("ai.json").startswith("нейросеть (DINOv2-small)")
        assert visual.EYE["random_cos"] == 0.62
        setup_ai.exam = lambda e: (hog, {"precision": 0.31, "random_cos": 0.62})
        setup_ai.main(["--dino"])                              # не лучше — остаётся HOG
        assert json.load(open("ai.json", encoding="utf-8"))["eye"] == "hog"
        setup_ai.main(["--dino", "--force"])
        assert json.load(open("ai.json", encoding="utf-8"))["eye"] == "neural"
        setup_ai.main(["--off"])
        assert json.load(open("ai.json", encoding="utf-8")) == {"eye": "hog"}
    finally:
        (setup_ai.install_onnxruntime, setup_ai.get_model, setup_ai.exam, visual.load_neural) = saved
        visual.use_hog()
        os.chdir(cwd)


def test_hog_works_without_opencv_hog():
    """HOG написан на numpy: у некоторых сборок OpenCV нет cv2.HOGDescriptor (ошибка у пользователя)."""
    import cv2
    saved = getattr(cv2, "HOGDescriptor", None)
    if saved is not None:
        del cv2.HOGDescriptor
    try:
        vector = visual.hog_describe(visual.normalize_ohlc(candles()))
        assert len(vector) == 1980 and abs(np.linalg.norm(vector) - 1) < 1e-6
    finally:
        if saved is not None:
            cv2.HOGDescriptor = saved


# --- своя нейросеть на графиках ---------------------------------------------------------------------
def fake_market(seed, n=1000):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.01))
    o = np.roll(c, 1)
    o[0] = c[0]
    w = np.abs(rng.normal(0, 0.004, n))
    v = rng.lognormal(0, 0.5, n)
    t0 = 1_780_000_000_000
    return [[t0 + i * 14_400_000, o[i], max(o[i], c[i]) * (1 + w[i]), min(o[i], c[i]) * (1 - w[i]), c[i], v[i]]
            for i in range(n)]


def test_chartnet_trains_saves_and_describes(tmp_path):
    import pytest
    pytest.importorskip("sklearn")                       # ставится командой python setup_ai.py
    import chartnet
    windows = eye_check.collect({f"S{i}": fake_market(i) for i in range(6)},
                                patterns=["Order Block", "Range", "Канал", "FVG"], per_pattern=60)
    model = chartnet.train(windows, max_iter=60)
    net = chartnet.ChartNet(model)
    vector = net(windows[0][1])
    assert len(vector) == chartnet.HIDDEN[-1] and abs(np.linalg.norm(vector) - 1) < 1e-5
    path = str(tmp_path / "net.npz")
    chartnet.save(model, path)
    again = chartnet.load(path)
    assert np.allclose(again(windows[0][1]), vector, atol=1e-5) and again.cases == len(windows)
    config = tmp_path / "ai.json"
    config.write_text(json.dumps({"eye": "chartnet", "model": path, "random_cos": 0.55}))
    try:
        assert visual.load_config(str(config)).startswith("своя нейросеть")
        assert visual.EYE["random_cos"] == 0.55 and len(visual.look(windows[1][1])) == chartnet.HIDDEN[-1]
    finally:
        visual.use_hog()


def test_setup_ai_trains_chartnet_and_decides(tmp_path):
    import os
    import pytest
    pytest.importorskip("sklearn")
    import setup_ai
    saved = (setup_ai.install_sklearn, setup_ai.MIN_TRAIN, setup_ai.EXAM_COINS, setup_ai.TRAIN_INTERVALS)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    setup_ai.install_sklearn = lambda: True
    setup_ai.MIN_TRAIN, setup_ai.EXAM_COINS, setup_ai.TRAIN_INTERVALS = 50, 3, ("4h",)
    fetch = lambda symbol, interval, limit: fake_market(hash((symbol, interval)) % 1000)
    try:
        data = setup_ai.load_data(fetch, [f"C{i}USDT" for i in range(8)])
        result = setup_ai.train_chartnet(data=data)
        assert result is not None
        model, hog, net = result
        assert model["cases"] >= 50 and 0 <= net["precision"] <= 1 and 0 <= hog["precision"] <= 1
        original = setup_ai.train_chartnet, setup_ai.load_data
        setup_ai.load_data = lambda: data
        setup_ai.train_chartnet = lambda data=None: (model, {**hog, "precision": 0.2}, {**net, "precision": 0.4})
        try:
            setup_ai.main([])
        finally:
            setup_ai.train_chartnet, setup_ai.load_data = original
        config = json.load(open("ai.json", encoding="utf-8"))
        assert config["eye"] == "chartnet" and os.path.exists(config["model"])
    finally:
        (setup_ai.install_sklearn, setup_ai.MIN_TRAIN, setup_ai.EXAM_COINS, setup_ai.TRAIN_INTERVALS) = saved
        visual.use_hog()
        os.chdir(cwd)


def test_personal_model_learns_what_you_like_by_view():
    """Дообучение на 👍/👎: нравятся «волны», не нравится «рост» — новая волна получает высокий шанс."""
    import feedback
    rng = np.random.default_rng(1)

    def frame(closes):
        return pd.DataFrame({"open": closes - 0.02, "high": closes + 0.05, "low": closes - 0.05, "close": closes})

    def wave():
        return frame(np.sin(np.linspace(0, 6, 80) + rng.normal(0, 0.2)) + rng.normal(0, 0.03, 80))

    def ramp():
        return frame(np.linspace(0, 1, 80) + rng.normal(0, 0.03, 80))

    item = {"start": 40, "window": 10}
    records = [{"name": "FVG", "vote": 1, "features": {}, "window": feedback.item_window(wave(), item)}
               for _ in range(6)]
    records += [{"name": "FVG", "vote": -1, "features": {}, "window": feedback.item_window(ramp(), item)}
                for _ in range(6)]
    assert feedback.personal_head(records[:8]) is None                     # мало оценок / нет обоих
    head = feedback.personal_head(records)
    assert head.votes == 12
    liked, disliked = head(feedback.item_window(wave(), item)), head(feedback.item_window(ramp(), item))
    assert liked > 0.7 and disliked < 0.3
    model = feedback.learn(records)
    assert model["FVG"]["head"] is not None
    assert "запомнил, как выглядят находки" in feedback.summary(records)
