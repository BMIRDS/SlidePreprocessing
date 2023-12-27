import numpy as np

def get_topk_threshold(arr, k):
    arr = arr.reshape(-1)
    if k >= len(arr):
        return arr.min()
    temp = np.argpartition(-arr, k)
    temp = np.partition(-arr, k)
    result = -temp[:k]
    return result.min()