import numpy as np

from shape import normalize, resample


def find_best_match(prices, shape, window_len, recent=None):
    best_start = 0
    best_score = -1
    first_start = 0
    if recent is not None:
        first_start = max(0, len(prices) - window_len - (recent - 1))
    for start in range(first_start, len(prices) - window_len + 1):
        segment = prices[start : start + window_len]
        segment_shape = normalize(resample(segment, len(shape)))
        score = np.corrcoef(shape, segment_shape)[0, 1]
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_score


def find_best_match_multi(prices, shape, windows, recent=None):
    best_start = 0
    best_score = -1
    best_window = 0
    for window_len in windows:
        start, score = find_best_match(prices, shape, window_len, recent)
        if score > best_score:
            best_score = score
            best_start = start
            best_window = window_len
    return best_start, best_score, best_window
