"""Включить ИИ: нейросеть-«глаз» и ИИ-значимость. Запуск из папки pattern_finder (бот можно
не останавливать — запускай во ВТОРОМ терминале):

    python setup_ai.py

Что делает (10–20 минут, один раз; потом можно повторять раз в месяц — данные свежие):
  1) ставит scikit-learn — библиотеку машинного обучения (только для обучения);
  2) качает историю Binance: 45 монет для обучения и 15 ДРУГИХ монет для экзамена;
  3) находит там тысячи настоящих паттернов и ОБУЧАЕТ свою нейросеть узнавать их форму
     (chartnet.py) — веса сохраняются в models/chartnet.npz;
  4) ЭКЗАМЕН на монетах, которых сеть не видела: кто лучше узнаёт похожие паттерны —
     сеть или обычный HOG (eye_check.py); решение — в ai.json;
  5) ИИ-ЗНАЧИМОСТЬ (quality.py): для каждого FVG, OB, слома, уровня ликвидности и фигуры
     на истории смотрит, дал ли он реакцию, и учится отличать значимые от шума.
     Экзамен — на других монетах и на последних по времени свечах; модель в models/quality.json;
  6) итог. Бот читает всё при запуске — перезапусти бота после этой команды.

    python setup_ai.py --force   — включить сеть, даже если она не лучше
    python setup_ai.py --off     — выключить сеть (вернуть HOG)
    python setup_ai.py --dino    — проверить готовую нейросеть DINOv2 (в первый раз она проиграла HOG)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

MODELS = [   # (название, адрес, файл) — пробуем по очереди
    ("DINOv2-small", "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx",
     "dinov2_small_quantized.onnx"),
    ("DINOv2-small", "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx",
     "dinov2_small.onnx"),
    ("MobileNetV2", "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/"
     "model/mobilenetv2-12.onnx", "mobilenetv2-12.onnx"),
]
MODELS_DIR = "models"
EXAM_COINS = 15
TRAIN_COINS = 45
EXAM_INTERVALS = ("1h", "4h", "1d")
TRAIN_INTERVALS = ("1h", "4h", "1d")
TRAIN_PER_PATTERN = 400
CHARTNET_PATH = os.path.join("models", "chartnet.npz")
QUALITY_PATH = os.path.join("models", "quality.json")
MIN_TRAIN, MIN_EXAM = 200, 30      # меньше случаев — данных мало, учить и экзаменовать нечестно


def say(text):
    print(text, flush=True)


def install_onnxruntime():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        pass
    say("1/5 Ставлю onnxruntime (программа для запуска нейросетей)…")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "onnxruntime"])
    if code != 0:
        say("❌ Не получилось поставить onnxruntime. Попробуй вручную:  pip install onnxruntime")
        return False
    return True


def download(url, path):
    """Скачивает файл с полоской прогресса. True — скачалось."""
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {done / 2**20:.0f} из {total / 2**20:.0f} МБ", end="", flush=True)
        print()
        os.replace(tmp, path)
        return os.path.getsize(path) > 1_000_000
    except Exception as error:
        say(f"\n   не скачалось: {error}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def get_model():
    """Скачивает первую рабочую модель и проверяет её. Возвращает (название, путь, глаз) или None."""
    import numpy as np
    import visual
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url, filename in MODELS:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            say(f"2/5 Скачиваю нейросеть {name}…")
            if not download(url, path):
                continue
        say(f"3/5 Проверяю {name}…")
        try:
            eye = visual.load_neural(path)
            started = time.time()
            vector = eye(np.random.default_rng(0).random((60, 4)))
            say(f"   работает: описание из {len(vector)} чисел за {time.time() - started:.2f} с")
            return name, path, eye
        except Exception as error:
            say(f"   эта модель не запустилась ({error}), пробую следующую")
    return None


def load_market(symbols, intervals, fetch=None):
    """Свечи {(монета, ТФ): свечи} для всех монет и таймфреймов."""
    from market import fetch_many, get_candles
    candles = {}
    for interval in intervals:
        for symbol, data in fetch_many(symbols, interval, 1000, fetch or get_candles).items():
            candles[(symbol, interval)] = data
    return candles


def exam_windows(fetch=None, symbols=None):
    import eye_check
    from market import top_symbols
    symbols = symbols or top_symbols(EXAM_COINS)
    return eye_check.collect(load_market(symbols, EXAM_INTERVALS, fetch))


def exam(eye, windows=None):
    """Экзамен HOG против другого глаза на реальном рынке."""
    import eye_check
    import visual
    if windows is None:
        say(f"Экзамен на рынке: {EXAM_COINS} монет, {', '.join(EXAM_INTERVALS)} (пара минут)…")
        windows = exam_windows()
    if len(windows) < 30:
        say("❌ Мало данных для экзамена (нет связи с Binance?). Попробуй позже.")
        return None
    hog = eye_check.evaluate(windows, visual.hog_describe)
    other = eye_check.evaluate(windows, eye)
    say(f"   случаев: {hog['cases']}, паттернов: {hog['patterns']}, угадывание наугад: {hog['chance']:.0%}")
    say(f"   HOG:        точность@5 {hog['precision']:.0%}")
    say(f"   нейросеть:  точность@5 {other['precision']:.0%}")
    return hog, other


def install_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        pass
    say("1/6 Ставлю scikit-learn (библиотека машинного обучения)…")
    if subprocess.call([sys.executable, "-m", "pip", "install", "scikit-learn"]) != 0:
        say("❌ Не получилось поставить scikit-learn. Попробуй вручную:  pip install scikit-learn")
        return False
    return True


def load_data(fetch=None, symbols=None):
    """История для обучения и экзамена: (свечи монет для обучения, свечи ДРУГИХ монет для экзамена)."""
    from market import top_symbols
    symbols = symbols or top_symbols(EXAM_COINS + TRAIN_COINS)
    exam_symbols, train_symbols = symbols[:EXAM_COINS], symbols[EXAM_COINS:]
    say(f"2/6 Качаю историю: {len(train_symbols)} монет для обучения ({', '.join(TRAIN_INTERVALS)}) "
        f"и {len(exam_symbols)} других для экзамена…")
    return load_market(train_symbols, TRAIN_INTERVALS, fetch), load_market(exam_symbols, EXAM_INTERVALS, fetch)


def train_chartnet(fetch=None, symbols=None, data=None):
    """Шаги 2–4: данные, обучение, экзамен. Возвращает (модель, HOG, сеть) или None."""
    import chartnet
    import eye_check
    train_candles, exam_candles = data or load_data(fetch, symbols)
    windows = eye_check.collect(train_candles, chartnet.TRAIN_PATTERNS, TRAIN_PER_PATTERN)
    exam_set = eye_check.collect(exam_candles)
    if len(windows) < MIN_TRAIN or len(exam_set) < MIN_EXAM:
        say("❌ Мало данных (нет связи с Binance?). Попробуй позже.")
        return None
    say(f"3/6 Учу нейросеть-«глаз» на {len(windows)} настоящих паттернах (несколько минут)…")
    started = time.time()
    model = chartnet.train(windows)
    say(f"   готово за {time.time() - started:.0f} с, угадывает паттерн в {model['accuracy']:.0%} случаев")
    say("4/6 Экзамен на монетах, которых сеть не видела:")
    result = exam(chartnet.ChartNet(model), exam_set)
    if result is None:
        return None
    return (model, *result)


def train_quality(data, net=None):
    """Шаг 5: ИИ-значимость — учится на ИСХОДАХ: какие паттерны на истории дали реакцию."""
    import quality
    from market import candles_to_df
    frames = lambda candles: [candles_to_df(c) for c in candles.values() if c and len(c) >= 300]
    train_frames, exam_frames = frames(data[0]), frames(data[1])
    if len(train_frames) < 10 or len(exam_frames) < 5:
        say("   мало данных для значимости — пропускаю")
        return None
    say(f"5/6 Учу ИИ-значимость: что из найденного на {len(train_frames)} графиках реально дало реакцию, "
        f"экзамен — на {len(exam_frames)} графиках других монет (последняя половина времени)…")
    started = time.time()
    models = quality.train_all(train_frames, exam_frames, net=net, say=say)
    say(f"   готово за {time.time() - started:.0f} с")
    if models:
        quality.save(models, QUALITY_PATH)
    return models


def save(config):
    with open("ai.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main(args):
    if "--off" in args:
        save({"eye": "hog"})
        say("Нейросеть выключена. Перезапусти бота.")
        return
    if "--dino" in args:
        main_dino(args)
        return
    if not install_sklearn():
        return
    data = load_data()
    result = train_chartnet(data=data)
    if result is None:
        return
    import chartnet
    import eye_check
    model, hog, net = result
    os.makedirs(MODELS_DIR, exist_ok=True)
    chartnet.save(model, CHARTNET_PATH)
    better = eye_check.decide(hog, net)
    use = better or "--force" in args
    save({"eye": "chartnet" if use else "hog", "model": CHARTNET_PATH, "random_cos": net["random_cos"],
          "exam": {"hog": hog, "chartnet": net}, "checked_at": time.strftime("%Y-%m-%d %H:%M")})
    if better:
        say("   ✅ Своя нейросеть включена — она узнаёт похожие паттерны лучше HOG.")
    elif use:
        say("   Своя нейросеть включена принудительно.")
    else:
        say("   Своя нейросеть пока НЕ лучше HOG — оставляю HOG (python setup_ai.py --force включит её).")
    models = train_quality(data, chartnet.ChartNet(model)) or {}
    on = [f for f, m in models.items() if m.get("on")]
    import quality
    if on:
        say("6/6 ✅ ИИ-значимость включена для: " + ", ".join(quality.FAMILY_TEXT[f] for f in on)
            + ". Перезапусти бота.")
    else:
        say("6/6 Значимость по истории пока не включилась (мало уверенности) — силу находок оценивает "
            "эксперт по критериям ТА и Smart Money, он работает и без неё. Перезапусти бота. "
            "Пришли этот вывод — разберёмся.")


def main_dino(args):
    """Готовая нейросеть DINOv2 (ONNX) — как в обновлении 21."""
    if not install_onnxruntime():
        return
    model = get_model()
    if model is None:
        say("❌ Не удалось скачать ни одну модель. Проверь интернет и запусти ещё раз.")
        return
    name, path, eye = model
    result = exam(eye)
    if result is None and "--force" not in args:
        return
    hog, neural = result if result else (None, None)
    import eye_check
    better = bool(result) and eye_check.decide(hog, neural)
    use = better or "--force" in args
    save({"eye": "neural" if use else "hog", "model": path, "model_name": name,
          "random_cos": neural["random_cos"] if neural else 0.5,
          "exam": {"hog": hog, "neural": neural}, "checked_at": time.strftime("%Y-%m-%d %H:%M")})
    if better:
        say(f"✅ Нейросеть {name} включена — она лучше HOG. Перезапусти бота.")
    elif use:
        say(f"Нейросеть {name} включена принудительно. Перезапусти бота.")
    else:
        say("Нейросеть НЕ лучше HOG — оставляю HOG.")


if __name__ == "__main__":
    main(sys.argv[1:])
