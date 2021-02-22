import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit


@njit
def distance(p1, p2):
    sum_sq = np.sum(np.square(point1 - point2))
    return np.sqrt(sum_sq)