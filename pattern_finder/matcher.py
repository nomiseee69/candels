import numpy as np

from shape import normalize, resample


def find_best_match(prices, shape, window_len):
    best_start = 0
    best_score = -1
    for start in range(len(prices) - window_len + 1):
        segment = prices[start : start + window_len]
        segment_shape = normalize(resample(segment, len(shape)))
        score = np.corrcoef(shape, segment_shape)[0, 1]
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_score

def find_best_match_multi(prices, shape, windows):
    best_start = 0
    best_score = -1
    best_window = 0
    for window_len in windows:
        start, score = find_best_match(prices, shape,window_len)
        if score > best_score:
            best_score = score
            best_start = start
            best_window = window_len
    return best_start, best_score, best_window