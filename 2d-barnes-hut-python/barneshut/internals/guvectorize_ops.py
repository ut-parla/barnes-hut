import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit


@njit
def gu_distance(p1, p2):
    # TODO: I'm not sure this works.. np.square is a ufunc, but I'm not sure
    # I can call it here since we are in another ufunc. Do we need to 
    # loop over all axis of each point?
    sum_sq = np.sum(np.square(p1 - p2))
    return np.sqrt(sum_sq)


# Inputs are:
# - The coordinates of a point (1d array)
# - A list of points (2d matrix)
# 
@guvectorize(["void(float64[:], float64[:,:],  float64[:])"], 
             '(n),(m, n)->(n)',nopython=False)
def gu_point_to_cloud(point, cloud, result):
    # AFAIK, the cool part of guvectorize is that if this function is called with a matrix of points
    # instead of a single point, it will be vectorized and possibly accelerated


    # need to figure out exactly what to do, mainly:
    # calculate distance from point to all points in cloud
    # calculate forces
    # calculate accelerations, return array so that caller can add to point's acceleration

    pass