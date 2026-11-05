"""Поставить «мозг» бота — свою языковую нейросеть Qwen2.5-3B (≈2 ГБ). Один раз, из папки
pattern_finder, во ВТОРОМ терминале (бот можно не останавливать, но потом перезапустить):

    python setup_brain.py            — поставить (сам выберет способ)
    python setup_brain.py --ollama   — через программу Ollama (если первый способ не встал)
    python setup_brain.py --test     — проверить, как мозг понимает фразы
    python setup_brain.py --off      — выключить мозг (бот вернётся к словарю)

Что делает:
  1) ставит пакет llama-cpp-python (запускает нейросети формата GGUF на процессоре);
  2) качает модель qwen2.5-3b-instruct-q4_k_m.gguf в папку models (≈2 ГБ; если связь оборвётся —
     запусти ещё раз, докачает с места обрыва);
  3) проверяет на трёх фразах и пишет brain.json — бот читает его при запуске.
Если пакет не ставится (бывает на Windows), запасной путь — Ollama: скачай с ollama.com,
установи, потом  python setup_brain.py --ollama  (модель скачает сама Ollama).
"""

import os
import subprocess
import sys
import time
import urllib.request

import brain

WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
TESTS = [
    ("BOS", "должен определяться BOS только у последней свечи которую если пересечь то меняет структура рынка"),
    ("Order Block", "ОБ, с которого ушли с отрывом, и чтоб до этого выбило стопы под лоями, объём жирный"),
    ("FVG", "имба, которую ещё не перекрыли, рядом с ценой, по тренду, на старших тф"),
]


def say(text):
    print(text, flush=True)


def install_llama():
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        pass
    say("1/3 Ставлю llama-cpp-python (готовую сборку для процессора)…")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--prefer-binary",
                            "--only-binary=llama-cpp-python", "--extra-index-url", WHEELS])
    if code != 0:
        say("❌ Готовой сборки llama-cpp-python для твоего Python не нашлось.\n"
            "   Запасной путь (проще): скачай и установи Ollama с https://ollama.com/download ,\n"
            "   потом выполни:  python setup_brain.py --ollama")
        return False
    return True


def download(url, path):
    """Качает большой файл с докачкой: оборвалась связь — запусти ещё раз, продолжит с места обрыва."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "ab" if done else "wb") as f:
            if done and response.status != 206:            # сервер не умеет докачку — с начала
                f.seek(0)
                f.truncate()
                done = 0
            total = done + int(response.headers.get("Content-Length") or 0)
            started = time.time()
            while True:
                chunk = response.read(4 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                print(f"\r   {done / 2**20:.0f} из {total / 2**20:.0f} МБ", end="", flush=True)
        print()
        os.replace(tmp, path)
        return os.path.getsize(path) > 500_000_000
    except Exception as error:
        say(f"\n   связь оборвалась: {error}. Запусти команду ещё раз — докачаю.")
        return False


def check():
    """Три фразы — как мозг их понял."""
    import language
    say("3/3 Проверяю, как я понимаю фразы (первый ответ — дольше: модель загружается в память)…")
    for name, text in TESTS:
        started = time.time()
        spec = brain.understand(text, name)
        if spec is None:
            say("❌ Нейросеть не ответила. Пришли этот вывод — разберёмся.")
            return False
        say(f"\n   «{text}» ({name}, {time.time() - started:.0f} с)\n   🧠 {spec.get('explain', '')}")
        for line in language.describe(spec):
            say(f"   • {line}")
    return True


def setup_llama():
    if not install_llama():
        return False
    if not os.path.exists(brain.MODEL_FILE):
        say("2/3 Качаю модель Qwen2.5-3B (≈2 ГБ, один раз)…")
        if not download(brain.MODEL_URL, brain.MODEL_FILE):
            return False
    else:
        say("2/3 Модель уже скачана.")
    brain.save_config({"backend": "llama", "model": brain.MODEL_FILE, "checked_at": time.strftime("%Y-%m-%d %H:%M")})
    return True


def setup_ollama():
    import requests
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        say("❌ Ollama не запущена. Установи её с https://ollama.com/download (после установки она "
            "работает в фоне, значок у часов), потом снова:  python setup_brain.py --ollama")
        return False
    say(f"1/3 Ollama работает. 2/3 Скачиваю модель {brain.OLLAMA_MODEL} (≈2 ГБ, один раз)…")
    try:
        with requests.post("http://127.0.0.1:11434/api/pull", json={"model": brain.OLLAMA_MODEL},
                           stream=True, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if b'"completed"' in line and b'"total"' in line:
                    import json
                    step = json.loads(line)
                    print(f"\r   {step['completed'] / 2**20:.0f} из {step['total'] / 2**20:.0f} МБ", end="",
                          flush=True)
        print()
    except requests.RequestException as error:
        say(f"❌ Не скачалось: {error}. Запусти ещё раз.")
        return False
    brain.save_config({"backend": "ollama", "model": brain.OLLAMA_MODEL,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M")})
    return True


def main(args):
    if "--off" in args:
        brain.save_config({})
        say("Мозг выключен — понимаю описания по словарю. Перезапусти бота.")
        return
    if "--test" not in args:
        ok = setup_ollama() if "--ollama" in args else setup_llama()
        if not ok:
            return
    if not brain.available():
        say("Мозг ещё не поставлен:  python setup_brain.py")
        return
    if check():
        say("\n✅ Мозг включён. Перезапусти бота — в консоли будет «Мозг (понимание слов): Qwen2.5-3B».\n"
            "Теперь можно писать боту своими словами: описание паттерна, вопрос, «просканируй рынок».")


if __name__ == "__main__":
    main(sys.argv[1:])
