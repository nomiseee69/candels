"""Мозг бота (brain.py): своя нейросеть понимает любые слова. Настоящей модели в тестах нет —
подменяем её ответы и проверяем всё вокруг: схему, чистку ответа, слияние со словарём, чат."""

import json

import brain
import language


class FakeBrain:
    """Вместо Qwen: отвечает заготовками и запоминает, что у неё спросили."""

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []
        self.saved = brain.load_config, brain.chat, brain._cache.copy(), brain._llama_ready
        brain.load_config = lambda: {"backend": "llama"}
        brain._llama_ready = lambda: True
        brain.chat = self.chat
        brain._cache.clear()

    def chat(self, messages, schema=None, max_tokens=500, temperature=0.1):
        self.asked.append((messages, schema))
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)

    def restore(self):
        brain.load_config, brain.chat, brain._llama_ready = self.saved[0], self.saved[1], self.saved[3]
        brain._cache.clear()
        brain._cache.update(self.saved[2])


def spec_answer(**kw):
    base = {"explain": "понял", "require": [], "prefer": [], "forbid": [], "timeframes": [], "direction": "any",
            "min_touches": 0, "unknown": []}
    return base | kw


def test_without_model_bot_uses_dictionary():
    assert brain.understand("OB с объёмом", "Order Block") is None or brain.available()
    saved = brain.load_config
    brain.load_config = lambda: {}
    try:
        assert not brain.available() and "словарю" in brain.status()
        assert language.understand("OB с объёмом", "Order Block")["require"] == ["volume"]
    finally:
        brain.load_config = saved


def test_schema_allows_only_conditions_of_this_pattern():
    lines = brain.concept_lines(language.GROUPS["BOS"])
    keys = [k for k, _ in lines]
    assert "key" in keys and "imbalance" not in keys            # у BOS нет «FVG внутри»
    schema = brain.spec_schema(keys)
    assert schema["properties"]["require"]["items"]["enum"] == keys
    messages = brain.spec_messages("последний bos", "BOS", lines)
    assert messages[0]["role"] == "system" and "key —" in messages[0]["content"]
    assert messages[-1]["content"].endswith("последний bos")


def test_model_answer_is_cleaned():
    keys = ["liquidity", "volume", "imbalance"]
    spec = brain.clean_spec(spec_answer(require=["liquidity", "magic", "volume"], forbid=["volume"],
                                        timeframes=["4h", "2h"], direction="down", min_touches=0,
                                        unknown=["цвет"]), keys)
    assert spec["require"] == ["liquidity"] and spec["forbid"] == ["volume"]      # «нельзя» важнее
    assert spec["timeframes"] == ["4h"] and spec["direction"] == "down" and spec["min_touches"] is None
    assert spec["unknown"] == ["цвет"] and spec["explain"] == "понял"


def test_any_wording_is_understood_with_model():
    fake = FakeBrain([spec_answer(explain="OB после выноса стопов, с отрывом", require=["liquidity", "impulse"])])
    try:
        text = "ОБ, с которого ушли с отрывом, и чтоб до этого выбило стопы под лоями, а ещё жирный объём"
        spec = language.understand(text, "Order Block")
        # нейросеть поняла главное, словарь добавил то, что она упустила (объём)
        assert {"liquidity", "impulse", "volume"} <= set(spec["require"]) and spec["brain"]
        assert spec["explain"].startswith("OB после")
        assert language.understand(text, "Order Block")["require"]      # второй раз — из памяти, без модели
        assert len(fake.asked) == 1
    finally:
        fake.restore()


def test_broken_model_answer_falls_back_to_dictionary():
    fake = FakeBrain(["я не знаю"])
    try:
        assert language.understand("OB с объёмом", "Order Block")["require"] == ["volume"]
    finally:
        fake.restore()


def chat_bot(tmp_path):
    import bot
    import storage
    from test_menus import Recorder, use_storage
    rec = Recorder()
    original = use_storage(tmp_path)
    pid = storage.add_pattern(5, "Order Block", None, note="без скриншота", builtin=True)
    storage.add_pattern(5, "Order Block", None, note="второй", builtin=True)
    return bot, storage, rec, original, pid


def test_free_text_description_goes_to_the_pattern(tmp_path):
    bot, storage, rec, original, pid = chat_bot(tmp_path)
    # паттерн назван и словарь понял условия — нейросеть не спрашиваем «что ты хочешь», сразу описание
    fake = FakeBrain([spec_answer(explain="OB со свипом", require=["liquidity"])])
    try:
        bot.handle_message({"chat": {"id": 5}, "text": "хочу чтобы мои ордерблоки были только после свипа"})
        assert len(fake.asked) == 1
        assert "Запомнил для «Order Block» (2 примера)" in rec.sent[-1][0] and "🧠 OB со свипом" in rec.sent[-1][0]
        assert all("liquidity" in p["spec"]["require"] for p in storage.get_patterns(5))
    finally:
        fake.restore()
        rec.restore()
        storage.FILE = original


def test_free_text_question_gets_an_answer(tmp_path):
    bot, storage, rec, original, _ = chat_bot(tmp_path)
    fake = FakeBrain([{"intent": "question", "pattern": "none", "coin": ""},
                      "Order Block — последняя свеча перед сильным движением."])
    try:
        bot.handle_message({"chat": {"id": 5}, "text": "что такое ордер блок простыми словами?"})
        assert rec.sent[-1][0].startswith("🧠 Order Block —")
        assert fake.asked[1][1] is None                          # ответ текстом, без схемы
    finally:
        fake.restore()
        rec.restore()
        storage.FILE = original


def test_free_text_without_model_uses_dictionary(tmp_path):
    bot, storage, rec, original, _ = chat_bot(tmp_path)
    saved = brain.load_config
    brain.load_config = lambda: {}
    try:
        bot.handle_message({"chat": {"id": 5}, "text": "ОБ только с объёмом"})
        assert "Запомнил для «Order Block»" in rec.sent[-1][0]
        bot.handle_message({"chat": {"id": 5}, "text": "как дела"})
        assert "Запомнил" not in rec.sent[-1][0]                # просто меню
    finally:
        brain.load_config = saved
        rec.restore()
        storage.FILE = original


def test_brain_json_without_package_or_model_means_dictionary(tmp_path):
    """brain.json скопировали на сервер, а модель там не поставили — бот честно работает по словарю,
    а не пытается каждый раз запустить несуществующую нейросеть."""
    saved = brain.load_config
    try:
        brain.load_config = lambda: {"backend": "llama", "model": str(tmp_path / "нет.gguf")}
        assert brain.backend() is None and not brain.available()
        assert "словар" in brain.status()
    finally:
        brain.load_config = saved



def test_model_mistakes_are_corrected_by_dictionary():
    """Ошибки маленькой модели, замеченные на сервере: «стопы под лоями» = бычий, а она сказала медвежий;
    у FVG «имбаланс внутри» бессмыслен; «не перекрыли» = свежая; «крупная» — она додумала."""
    fake = FakeBrain([spec_answer(explain="Медвежий OB", require=["liquidity", "volume"], direction="down"),
                      spec_answer(explain="FVG", require=["imbalance"], prefer=["near_price", "trend", "big"])])
    try:
        ob = language.understand("ОБ, с которого ушли с отрывом, и чтоб до этого выбило стопы под лоями, объём жирный",
                                 "Order Block")
        assert ob["direction"] == "up" and "direction" in ob["doubts"]
        fvg = language.understand("имба, которую ещё не перекрыли, рядом с ценой, по тренду, на старших тф", "FVG")
        assert "imbalance" not in fvg["require"] + fvg["prefer"] and "fresh" in fvg["require"]
        assert "big" in fvg["guessed"] and "fresh" not in fvg["guessed"]
        assert fvg["timeframes"] == ["12h", "1d", "1w"]
    finally:
        fake.restore()


def test_user_fixes_what_bot_understood_with_buttons(tmp_path):
    """Под «понял так» — кнопки: убрать лишнее условие, направление наоборот, всё верно."""
    bot, storage, rec, original, pid = chat_bot(tmp_path)
    saved = brain.load_config
    brain.load_config = lambda: {}
    try:
        bot.receive_description(5, pid, "бычий ОБ со снятием ликвидности и объёмом, на 4ч")
        text, keyboard = rec.sent[-1][0], rec.sent[-1][1]
        datas = [b["callback_data"] for row in keyboard["inline_keyboard"] for b in row]
        assert "Всё верно?" in text and f"spok:{pid}" in datas and f"spflip:{pid}" in datas
        assert f"sprm:{pid}:volume" in datas and f"sprm:{pid}:tf" in datas
        press = lambda data: bot.handle_callback({"id": "q", "data": data, "from": {"id": 5},
                                                  "message": {"chat": {"id": 5}, "message_id": 77}})
        press(f"sprm:{pid}:volume")
        spec = storage.get_pattern(5, pid)["spec"]
        assert "volume" not in spec["require"] and "liquidity" in spec["require"]
        press(f"spflip:{pid}")
        assert storage.get_pattern(5, pid)["spec"]["direction"] == "down"
        press(f"sprm:{pid}:tf")
        assert storage.get_pattern(5, pid)["spec"]["timeframes"] == []
        press(f"spok:{pid}")
    finally:
        brain.load_config = saved
        rec.restore()
        storage.FILE = original


def test_screenshot_direction_can_be_flipped(tmp_path):
    """Бот понял направление на скрине наоборот — «🔄 Нет, это медвежий» меняет пример и его отпечаток."""
    bot, storage, rec, original, _ = chat_bot(tmp_path)
    try:
        pid = storage.add_pattern(5, "FVG", None, note="скрин", kind="bullish_fvg",
                                  fingerprint={"family": "fvg", "kind": "bullish_fvg", "bullish": True})
        keyboard = bot.example_keyboard(pid, {"kind": "bullish_fvg"})
        assert any(b.get("callback_data") == f"flipdir:{pid}" and "медвежий" in b["text"]
                   for row in keyboard["inline_keyboard"] for b in row)
        assert not any("flipdir" in (b.get("callback_data") or "") for row in
                       bot.example_keyboard(pid, {"kind": "bos"})["inline_keyboard"] for b in row)
        bot.handle_callback({"id": "q", "data": f"flipdir:{pid}", "from": {"id": 5},
                             "message": {"chat": {"id": 5}, "message_id": 9}})
        p = storage.get_pattern(5, pid)
        assert p["kind"] == "bearish_fvg" and p["fingerprint"]["kind"] == "bearish_fvg"
        assert p["fingerprint"]["bullish"] is False and "медвежий" in rec.sent[-1][0]
    finally:
        rec.restore()
        storage.FILE = original
