import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit
from math import sqrt

# TODO: this won't work with CUDA because it requires only one return value and I can't figure
# out how to do so. We could return it as the last element of cloud_accels, but then the reduction
# yields the wrong result.
@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64[:], float64[:,:]"], 
             '(d),(), (n,d), (n), () -> (d), (n,d)', nopython=True, target="cpu")
def guvect_point_to_cloud(p_pos, p_mass, cloud_positions, cloud_masses, G, p_accel, cloud_accels):
    # AFAIK, the cool part of guvectorize is that if this function is called with a matrix of points
    # instead of a single point, it will be vectorized and possibly accelerated

    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    p_accel[:] = 0.0

    n = cloud_positions.shape[0]
    # TODO: figure out if we can do array calculations here instead of a loop
    for i in range(n):
        dif = p_pos-cloud_positions[i]
        dist = np.sqrt(np.sum(np.square(dif)))

        f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
        p_accel         -= (f * dif / p_mass)
        cloud_accels[i] += (f * dif / cloud_masses[i])


@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64[:,:]"], 
             '(d),(), (n,d), (n), () -> (n,d)', nopython=True, target="cuda")
def guvect_point_to_cloud_v2(p_pos, p_mass, cloud_positions, cloud_masses, G, cloud_accels):
    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    n = cloud_positions.shape[0]-1
    p_i = n

    for i in range(n):
        x, y = p_pos[0] - cloud_positions[i][0], p_pos[1] - cloud_positions[i][1]
        
        dist = sqrt((x*x) + (y*y))

        f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
        cloud_accels[p_i][0] -= (f * x / p_mass)
        cloud_accels[p_i][1] -= (f * y / p_mass)
        
        cloud_accels[i][0] += (f * x / cloud_masses[i])
        cloud_accels[i][1] += (f * y / cloud_masses[i])


    # cloud_accels[:, :] = 0.0
    # n = cloud_positions.shape[0]-1
    # p_i = n

    # for i in range(n):
    #     dif = p_pos - cloud_positions[i]
    #     dist = np.sqrt(np.sum(np.square(dif)))
    #     f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
    #     cloud_accels[p_i] -= (f * dif / p_mass)
    #     cloud_accels[i]   += (f * dif / cloud_masses[i])




        