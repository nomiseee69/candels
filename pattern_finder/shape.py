"""Превращает картинку пользователя в форму паттерна: n чисел от 0 до 1."""

import numpy as np

from vision import analyze_screenshot

SHAPE_POINTS = 50


def normalize(values):
    """Переводит любые числа в диапазон 0..1 (пиксели и доллары становятся сравнимы)."""
    arr = np.array(values, dtype=float)
    if arr.max() == arr.min():
        return np.zeros(len(arr))
    return (arr - arr.min()) / (arr.max() - arr.min())


def resample(values, n):
    """Сжимает или растягивает ряд до ровно n точек, сохраняя форму."""
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, n)
    return np.interp(new_x, old_x, values)


def photo_to_shape(path, n=SHAPE_POINTS):
    """Скриншот → форма. Бросает ValueError, если график не распознан (или картинки нет)."""
    if not path:
        raise ValueError("нет картинки")
    analysis = analyze_screenshot(path)
    if not analysis.ok:
        raise ValueError("Не удалось распознать график на картинке")
    return normalize(resample(analysis.pattern_series(), n))
