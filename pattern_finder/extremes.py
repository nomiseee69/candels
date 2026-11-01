def find_local_minima(values, window):
    minima = []
    for i in range(window, len(values) - window):
        left = values[i - window : i]
        right = values[i + 1 : i + 1 + window]
        if all(x > values[i] for x in left) and all(y > values[i] for y in right):
            minima.append(i)
    return minima


def find_local_maxima(values, window):
    maxima = []
    for i in range(window, len(values) - window):
        left = values[i - window : i]
        right = values[i + 1 : i + 1 + window]
        if max(left) < values[i] and max(right) < values[i]:
            maxima.append(i)
    return maxima
