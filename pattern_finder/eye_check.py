"""Экзамен для «глаза»: какой глаз (HOG или нейросеть) лучше узнаёт похожие паттерны.

Честная проверка на НАСТОЯЩЕМ рынке, а не на словах:
  1) на топ-монетах находим реальные случаи паттернов по правилам (Order Block, FVG,
     BOS, Range, флаг, треугольник, двойное дно, равные хаи/лои);
  2) каждый случай вырезаем с запасом (40 свечей до, 20 после) и рисуем картинкой;
  3) для каждой картинки глаз ищет 5 самых похожих среди всех остальных.
     Хороший глаз находит рядом случаи ТОГО ЖЕ паттерна: FVG рядом с FVG, Range рядом с Range.
     Доля таких попаданий — «точность@5». У случайного угадывания она ≈ 1 / число паттернов.
Нейросеть включаем, только если её точность заметно выше, чем у HOG.

Заодно меряем «случайную планку» глаза — обычный косинус между РАЗНЫМИ паттернами:
от неё считается «похоже на твой пример N%».
"""

import numpy as np

import visual
from fingerprint import CONTEXT_AFTER, CONTEXT_BEFORE
from market import candles_to_df
from scanner import build_history_jobs
from templates import template_shape

PATTERNS = ["Order Block", "FVG", "BOS", "Range", "Флаг", "Треугольник", "Двойное дно", "EQH / EQL"]
PER_PATTERN = 40           # не больше 40 случаев каждого паттерна — экзамен должен быть честным
TOP = 5
BETTER_BY = 0.03           # нейросеть должна быть лучше хотя бы на 3 процентных пункта


def is_bearish(item):
    kind = str(item.get("kind", ""))
    return kind.startswith("bearish") or item.get("direction") == "down" or kind == "eqh"


def windows_from_df(df, patterns=PATTERNS):
    """Все случаи паттернов на графике → [(паттерн, свечи 0..1)]. Медвежьи переворачиваем
    в бычьи — сравниваем форму, а не направление (так же делает сравнение с примером)."""
    jobs, _ = build_history_jobs([{"name": n, "shape": template_shape(n)} for n in patterns],
                                 {"fvg": "significant"})
    n = len(df)
    result = []
    for name, _, run in jobs:
        for item in run(df):
            start = int(item["start"])
            end = min(int(item.get("end", start + item.get("window", 1) - 1)), n - 1)
            lo, hi = max(0, start - CONTEXT_BEFORE), min(n, end + 1 + CONTEXT_AFTER)
            if hi - lo < 15:
                continue
            ohlc = visual.normalize_ohlc(df, lo, hi, mirror=is_bearish(item))
            if ohlc is not None:
                result.append((name, ohlc))
    return result


def collect(candles_by_symbol, patterns=PATTERNS, per_pattern=PER_PATTERN):
    """Случаи паттернов со многих монет, не больше per_pattern каждого (равномерно по списку)."""
    found = {}
    for candles in candles_by_symbol.values():
        if not candles or len(candles) < 150:
            continue
        for name, ohlc in windows_from_df(candles_to_df(candles), patterns):
            found.setdefault(name, []).append(ohlc)
    windows = []
    for name, items in found.items():
        if len(items) < TOP + 1:
            continue                                           # слишком мало случаев — не экзаменуем
        picks = np.linspace(0, len(items) - 1, min(per_pattern, len(items))).astype(int)
        windows += [(name, items[i]) for i in sorted(set(picks))]
    return windows


def evaluate(windows, describe, top=TOP):
    """Экзамен одного глаза: точность@top, «случайная планка» (медиана косинуса у разных
    паттернов) и медиана косинуса у одинаковых."""
    names = np.array([name for name, _ in windows])
    vectors = np.array([describe(ohlc) for _, ohlc in windows])
    cos = vectors @ vectors.T
    np.fill_diagonal(cos, -np.inf)
    hits = []
    for i in range(len(windows)):
        nearest = np.argsort(cos[i])[::-1][:top]
        hits.append(np.mean(names[nearest] == names[i]))
    same = names[:, None] == names[None, :]
    finite = np.isfinite(cos)
    return {
        "precision": float(np.mean(hits)),
        # угадывание наугад: доля того же паттерна среди ОСТАЛЬНЫХ случаев
        "chance": float(np.mean([(np.sum(names == n) - 1) / (len(names) - 1) for n in names])),
        "random_cos": float(np.median(cos[~same & finite])),
        "same_cos": float(np.median(cos[same & finite])),
        "cases": len(windows),
        "patterns": len(set(names)),
    }


def decide(hog, neural, better_by=BETTER_BY):
    """Включать ли нейросеть: только если она заметно точнее HOG."""
    return neural["precision"] >= hog["precision"] + better_by
