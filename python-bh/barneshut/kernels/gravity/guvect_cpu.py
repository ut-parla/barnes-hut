import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit
from math import sqrt
from functools import partial

# this is what is called from __init__.py
def get_kernel_function(target):
    if target == "cpu":
        return partial(guvect_cpu, guvect_point_to_cloud_cpu)
    elif target == "parallel":
        return partial(guvect_cpu, guvect_point_to_cloud_parallel)


def guvect_cpu(func, self_cloud, other_cloud, G):
    # this is really cool. the signature of this function takes a single point, but we pass multiple points instead
    # this way numpy can vectorize these operations.
    # let's do the biggest set as first parameter, since guvectorize parallelize based on it's shape
    if self_cloud.n >= other_cloud.n:
        c1, c2 = self_cloud, other_cloud
    else:
        c1, c2 = other_cloud, self_cloud
    is_self_self = 1.0 if c1 is c2 else 0.0

    print(f"c2 pos {c2.positions.shape}")
    print(f"c2 masses {c2.masses.shape}")
    c1_acc, c2_acc = func(c1.positions, c1.masses, c2.positions, c2.masses.squeeze(axis=1), G, is_self_self)
    c1.accelerations += c1_acc
    c2.accelerations += np.add.reduce(c2_acc)


@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64, float64[:], float64[:,:]"], 
             '(d),(), (n,d), (n), (), () -> (d), (n,d)', nopython=True, target="cpu", cache=True)
def guvect_point_to_cloud_cpu(p_pos, p_mass, cloud_positions, cloud_masses, G, is_self_self, p_accel, cloud_accels):
    # AFAIK, the cool part of guvectorize is that if this function is called with a matrix of points
    # instead of a single point, it will be vectorized and possibly accelerated

    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    p_accel[:] = 0.0

    n = cloud_positions.shape[0]
    for i in range(n):
        dif = p_pos-cloud_positions[i]
        dist = np.sqrt(np.sum(np.square(dif)))

        f = (G * p_mass * cloud_masses[i]) / (dist*dist)
        p_accel -= (f * dif / p_mass)
        if is_self_self != 0:
            cloud_accels[i] += (f * dif / cloud_masses[i])


@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64, float64[:], float64[:,:]"], 
             '(d),(), (n,d), (n), (), () -> (d), (n,d)', nopython=True, target="parallel", cache=True)
def guvect_point_to_cloud_parallel(p_pos, p_mass, cloud_positions, cloud_masses, G, is_self_self, p_accel, cloud_accels):
    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    p_accel[:] = 0.0

    n = cloud_positions.shape[0]
    # TODO: figure out if we can do array calculations here instead of a loop
    for i in range(n):
        dif = p_pos-cloud_positions[i]
        dist = np.sqrt(np.sum(np.square(dif)))

        f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
        p_accel -= (f * dif / p_mass)
        if is_self_self != 0:
            cloud_accels[i] += (f * dif / cloud_masses[i])
