heights = [5, 3, 4, 6, 2, 2.5, 7, 4, 1, 3, 5]

def find_local_minima(values, window):
    minima = []
    for i in range(window,len(values)-window):
        left = values[i-window:i]
        right= values[i+1:i+1+window]
        if all(x > values[i] for x in left) and all(y > values[i] for y in right):
            minima.append(i)
    return minima 
if __name__ == '__main__':
    print(find_local_minima(heights,2))
    print(find_local_minima(heights,1))
    test = [5, 6, 7, 1, 4]
    print(find_local_minima(test, 2))
        
    print(find_local_minima([1, 2, 3, 4, 5], 1))