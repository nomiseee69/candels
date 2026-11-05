"""Мозг бота (brain.py): своя нейросеть понимает любые слова. Настоящей модели в тестах нет —
подменяем её ответы и проверяем всё вокруг: схему, чистку ответа, слияние со словарём, чат."""

import json

import brain
import language


class FakeBrain:
    """Вместо Qwen: отвечает заготовками и запоминает, что у неё спросили."""

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []
        self.saved = brain.load_config, brain.chat, brain._cache.copy()
        brain.load_config = lambda: {"backend": "llama"}
        brain.chat = self.chat
        brain._cache.clear()

    def chat(self, messages, schema=None, max_tokens=500, temperature=0.1):
        self.asked.append((messages, schema))
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)

    def restore(self):
        brain.load_config, brain.chat = self.saved[0], self.saved[1]
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
    fake = FakeBrain([{"intent": "describe", "pattern": "Order Block", "coin": ""},
                      spec_answer(explain="OB со свипом", require=["liquidity"])])
    try:
        bot.handle_message({"chat": {"id": 5}, "text": "хочу чтобы мои ордерблоки были только после свипа"})
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
