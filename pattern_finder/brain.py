"""«Мозг» бота: своя языковая нейросеть прямо на твоём ПК, без интернета и внешних API.

Модель — Qwen2.5-3B-Instruct (≈2 ГБ, понимает русский и английский). Ставится один раз:

    python setup_brain.py

Зачем она нужна. Словарь в language.py знает только те слова и фразы, что в нём записаны.
Нейросеть понимает ЛЮБУЮ формулировку («ОБ, с которого ушли с отрывом, и чтоб до этого
выбило стопы под лоями») и переводит её в те же условия, что умеет проверять бот на свечах:
обязательно / желательно / нельзя, таймфреймы, направление, касания.

Как она отвечает строго по делу. Мы даём ей:
  1) роль: «ты — эксперт по SMC / ICT и классическому ТА»;
  2) список условий, которые бот умеет проверять для ЭТОГО паттерна (ключ и что значит);
  3) пару примеров «описание → ответ»;
  4) JSON-схему ответа. Схема превращается в «грамматику», и нейросеть физически не может
     написать ничего, кроме правильного JSON с ключами из списка (выдумать условие нельзя).

Два способа запустить модель (setup_brain.py сам выберет рабочий и запишет в brain.json):
  • "llama"  — пакет llama-cpp-python, модель в models/*.gguf, работает внутри бота;
  • "ollama" — программа Ollama (ollama.com), модель qwen2.5:3b, бот обращается к ней
               по адресу 127.0.0.1 (это твой же компьютер, не интернет).
Нет модели — бот работает как раньше, по словарю (language.parse).
"""

import hashlib
import json
import os
import threading
import time

CONFIG_PATH = "brain.json"
MODEL_FILE = os.path.join("models", "qwen2.5-3b-instruct-q4_k_m.gguf")
MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
             "qwen2.5-3b-instruct-q4_k_m.gguf")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
CONTEXT = 4096
TIMEOUT = 180                 # секунд на один ответ (первый раз модель ещё загружается в память)
TIMEFRAMES = ["1h", "4h", "12h", "1d", "1w"]

_lock = threading.Lock()      # нейросеть одна — думает над одним вопросом за раз
_llm = None
_cache = {}
_config = None


# --- какая модель включена -----------------------------------------------------------------------------
def load_config():
    """brain.json → {"backend": "llama" | "ollama", ...} или {} (мозга нет)."""
    global _config
    if _config is None:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                _config = json.load(f)
        except (OSError, ValueError):
            _config = {}
    return _config


def save_config(config):
    global _config, _llm
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    _config, _llm = config, None


def backend():
    """"llama", "ollama" или None. PF_BRAIN=off в .env — выключить, не удаляя модель."""
    if os.environ.get("PF_BRAIN", "").lower() in ("off", "0", "no"):
        return None
    name = load_config().get("backend")
    if name == "llama" and not _llama_ready():
        return None                      # brain.json есть, а пакета или модели нет (например, скопировали на сервер)
    return name if name in ("llama", "ollama") else None


def _llama_ready():
    """Пакет llama-cpp-python стоит и файл модели на месте."""
    import importlib.util
    if importlib.util.find_spec("llama_cpp") is None:
        return False
    return os.path.exists(load_config().get("model", MODEL_FILE).replace("\\", "/"))


def busy():
    """Нейросеть сейчас отвечает кому-то (следующий подождёт)."""
    return _lock.locked()


def available():
    return backend() is not None


def status():
    """Строка для консоли при запуске бота."""
    name = backend()
    if name == "llama":
        return "Qwen2.5-3B внутри бота (llama-cpp)"
    if name == "ollama":
        return f"Qwen2.5-3B через Ollama ({load_config().get('model', OLLAMA_MODEL)})"
    return "нет — описания понимаю по словарю (python setup_brain.py включит нейросеть)"


# --- разговор с моделью --------------------------------------------------------------------------------
def _llama():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        path = load_config().get("model", MODEL_FILE).replace("\\", "/")
        _llm = Llama(model_path=path, n_ctx=CONTEXT, n_threads=max(2, (os.cpu_count() or 4) - 1),
                     verbose=False)
    return _llm


def chat(messages, schema=None, max_tokens=500, temperature=0.1):
    """Сообщения [{"role", "content"}] → текст ответа модели. schema — JSON-схема ответа
    (модель обязана ответить ровно таким JSON). Мозга нет или ошибка — исключение."""
    name = backend()
    if name is None:
        raise RuntimeError("мозг не установлен")
    with _lock:
        if name == "llama":
            kwargs = {"response_format": {"type": "json_object", "schema": schema}} if schema else {}
            out = _llama().create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                  temperature=temperature, **kwargs)
            return out["choices"][0]["message"]["content"]
        import requests
        body = {"model": load_config().get("model", OLLAMA_MODEL), "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": CONTEXT}}
        if schema:
            body["format"] = schema
        response = requests.post(OLLAMA_URL, json=body, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]["content"]


def ask_json(messages, schema, max_tokens=500):
    """Ответ модели как словарь. Модель ответила не JSON — ValueError."""
    text = chat(messages, schema, max_tokens)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"не JSON: {text[:80]}")
    return json.loads(text[start:end + 1])


# --- описание паттерна → условия -----------------------------------------------------------------------
ROLE = ("Ты — опытный трейдер и эксперт по Smart Money Concepts (SMC / ICT) и классическому "
        "техническому анализу. Ты понимаешь трейдерский сленг: ликва, ОБ, имба, фвг, чох, бос, свип, "
        "сняли стопы, выбило, проторговка, хаи/лои, дисконт/премиум, HTF/LTF и т.п. "
        "Ты работаешь внутри бота, который ищет паттерны на графиках криптовалют.")

TASK = ("Пользователь описал своими словами, каким должен быть паттерн «{name}». Переведи описание "
        "в условия, которые бот умеет проверять на свечах. Список условий (ключ — смысл):\n{concepts}\n\n"
        "Правила:\n"
        "• require — без этого паттерн не нужен («обязательно», «должен», «только», или просто "
        "перечислено как свойство);\n"
        "• prefer — желательно («лучше», «желательно», «в идеале», «плюсом»);\n"
        "• forbid — нельзя («без», «не должно быть», «исключить»). Внимание: «не тестировался», "
        "«не заполнен», «не снята» — это свойства (свежая зона, ликвидность не снята), а не запрет;\n"
        "• timeframes — только если таймфреймы названы (1h 4h 12h 1d 1w; «старшие ТФ» = 12h 1d 1w);\n"
        "• direction — up (бычий, лонг, вверх), down (медвежий, шорт, вниз) или any;\n"
        "• min_touches — число касаний, если названо, иначе 0;\n"
        "• touches_each — true, если касаний столько на КАЖДОЙ линии («сверху и снизу», «на каждой границе»);\n"
        "• unknown — части описания, которые нельзя выразить условиями из списка (коротко);\n"
        "• explain — одной фразой по-русски, как ты понял описание.\n"
        "Используй только ключи из списка. Не добавляй условий, которых нет в описании.")

EXAMPLES = [
    ("BOS", "должен определяться BOS только у последней свечи которую если пересечь то меняет структуру рынка",
     {"explain": "Нужен только последний слом, пробой которого меняет структуру (тренд).",
      "require": ["key"], "prefer": [], "forbid": [], "timeframes": [], "direction": "any",
      "min_touches": 0, "touches_each": False, "unknown": []}),
    ("Order Block", "медвежий ОБ после сбора ликвы над хаями, с жирным объёмом, без имбы, лучше в премиуме, 4ч",
     {"explain": "Медвежий OB на 4H, который снял ликвидность, с повышенным объёмом, без FVG; "
                 "желательно в премиуме.",
      "require": ["liquidity", "volume"], "prefer": ["discount"], "forbid": ["imbalance"],
      "timeframes": ["4h"], "direction": "down", "min_touches": 0, "touches_each": False, "unknown": []}),
]


def concept_lines(group):
    import language
    return [(key, label) for key, (label, _, _, groups) in language.CONCEPTS.items()
            if group is None or group in groups]


def spec_schema(keys):
    concept = {"type": "string", "enum": keys}
    return {
        "type": "object",
        "properties": {
            "explain": {"type": "string"},
            "require": {"type": "array", "items": concept},
            "prefer": {"type": "array", "items": concept},
            "forbid": {"type": "array", "items": concept},
            "timeframes": {"type": "array", "items": {"type": "string", "enum": TIMEFRAMES}},
            "direction": {"type": "string", "enum": ["up", "down", "any"]},
            "min_touches": {"type": "integer"},
            "touches_each": {"type": "boolean"},
            "unknown": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["explain", "require", "prefer", "forbid", "timeframes", "direction", "min_touches",
                     "touches_each", "unknown"],
    }


def spec_messages(text, name, lines):
    concepts = "\n".join(f"- {key} — {label}" for key, label in lines)
    messages = [{"role": "system", "content": ROLE + "\n\n" + TASK.format(name=name, concepts=concepts)}]
    for ex_name, ex_text, ex_answer in EXAMPLES:
        messages.append({"role": "user", "content": f"Паттерн: {ex_name}\nОписание: {ex_text}"})
        messages.append({"role": "assistant", "content": json.dumps(ex_answer, ensure_ascii=False)})
    messages.append({"role": "user", "content": f"Паттерн: {name}\nОписание: {text}"})
    return messages


def clean_spec(answer, keys):
    """Ответ модели → описание в формате language.parse (лишнее и повторы выкидываем)."""
    spec = {"require": [], "prefer": [], "forbid": [], "timeframes": [], "direction": None,
            "min_touches": None, "understood": [], "unknown": [], "text": ""}
    seen = set()
    for mode in ("forbid", "require", "prefer"):          # «нельзя» важнее, если модель указала дважды
        for key in answer.get(mode) or []:
            if key in keys and key not in seen:
                seen.add(key)
                spec[mode].append(key)
                spec["understood"].append((key, mode))
    spec["timeframes"] = [tf for tf in TIMEFRAMES if tf in (answer.get("timeframes") or [])]
    if answer.get("direction") in ("up", "down"):
        spec["direction"] = answer["direction"]
    touches = answer.get("min_touches")
    if isinstance(touches, int) and 1 < touches < 50:
        spec["min_touches"] = touches
    spec["touches_each"] = bool(answer.get("touches_each"))
    spec["unknown"] = [str(u)[:80] for u in answer.get("unknown") or [] if str(u).strip()][:4]
    spec["explain"] = str(answer.get("explain") or "")[:300]
    return spec


def understand(text, name):
    """Описание словами → условия (как language.parse) или None, если мозга нет / он ошибся."""
    if not available() or not (text or "").strip():
        return None
    import language
    lines = concept_lines(language.GROUPS.get(name))
    keys = [k for k, _ in lines]
    key = hashlib.sha1(f"{name}|{text}".encode()).hexdigest()
    if key in _cache:
        return json.loads(json.dumps(_cache[key]))
    try:
        started = time.time()
        answer = ask_json(spec_messages(text, name, lines), spec_schema(keys))
        spec = clean_spec(answer, keys)
        spec["brain_seconds"] = round(time.time() - started, 1)
    except Exception as error:
        print("Мозг не смог разобрать описание:", error)
        return None
    _cache[key] = spec
    return json.loads(json.dumps(spec))


# --- свободный текст в чате: что хочет пользователь ----------------------------------------------------
INTENT_TASK = ("Пользователь написал боту сообщение. Определи, чего он хочет:\n"
               "• describe — описывает, каким должен быть паттерн (pattern — какой из списка);\n"
               "• question — задаёт вопрос о трейдинге, паттернах или о том, как работает бот;\n"
               "• scan — просит поискать / просканировать рынок сейчас;\n"
               "• coin — просит проверить конкретную монету (coin — тикер, например BTC);\n"
               "• learning — спрашивает, чему бот научился по оценкам;\n"
               "• menu — всё остальное (приветствие, непонятно).\n"
               "Паттерны: {patterns}.")


def intent_schema(patterns):
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["describe", "question", "scan", "coin", "learning", "menu"]},
            "pattern": {"type": "string", "enum": list(patterns) + ["none"]},
            "coin": {"type": "string"},
        },
        "required": ["intent", "pattern", "coin"],
    }


def intent(text, patterns):
    """Что хочет пользователь: {"intent", "pattern", "coin"} или None (мозга нет / ошибка)."""
    if not available():
        return None
    messages = [{"role": "system", "content": ROLE + "\n\n" + INTENT_TASK.format(patterns=", ".join(patterns))},
                {"role": "user", "content": text}]
    try:
        answer = ask_json(messages, intent_schema(patterns), max_tokens=80)
    except Exception as error:
        print("Мозг не понял сообщение:", error)
        return None
    if answer.get("pattern") not in patterns:
        answer["pattern"] = None
    return answer


ANSWER_TASK = ("Ответь пользователю по-русски, коротко (до 6 предложений), по делу, как наставник-трейдер. "
               "Объясняй понятия SMC / ICT и теханализа простыми словами. Не давай торговых сигналов и "
               "не обещай прибыль. Если вопрос о боте: бот ищет паттерны по скриншотам-примерам "
               "пользователя, по его описанию словами и учится на оценках 👍/👎; разметка — в TradingView.")


def answer(text):
    """Ответ на вопрос о трейдинге или None."""
    if not available():
        return None
    try:
        return chat([{"role": "system", "content": ROLE + "\n\n" + ANSWER_TASK},
                     {"role": "user", "content": text}], max_tokens=350, temperature=0.3).strip() or None
    except Exception as error:
        print("Мозг не смог ответить:", error)
        return None
