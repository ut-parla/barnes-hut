import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit


@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64[:], float64[:,:]"], 
             '(d),(), (n,d), (n), () -> (d), (n,d)', nopython=True, target="cpu")
def guvect_point_to_cloud(p_pos, p_mass, cloud_positions, cloud_masses, G, p_accel, cloud_accels):
    # AFAIK, the cool part of guvectorize is that if this function is called with a matrix of points
    # instead of a single point, it will be vectorized and possibly accelerated

    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    p_accel[:] = 0.0

    n = cloud_positions.shape[0]
    for i in range(n):
        dif = p_pos-cloud_positions[i]
        dist = np.sqrt(np.sum(np.square(dif)))

        f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
        p_accel         -= (f * dif / p_mass)
        cloud_accels[i] += (f * dif / cloud_masses[i])