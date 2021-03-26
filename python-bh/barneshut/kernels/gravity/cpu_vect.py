import numpy as np
from barneshut.internals import Cloud
from numba import njit
import barneshut.internals.particle as p

# calculations below are from this source:
# https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python

# this is what is called from __init__.py
def get_kernel_function():
    return cpu_vect

def cpu_vect(self_cloud, other_cloud, G, update_other=False):
    cpu_vect_kernel(self_cloud.particles, other_cloud.particles, G, 1.0 if update_other else 0.0)

@njit("(float64[:, :], float64[:, :], float64, float64)", fastmath=True)
def cpu_vect_kernel(self_cloud, other_cloud, G, update_other):
    # get positions and masses of concatenation
    for i1, p1 in enumerate(self_cloud):
        for i2, p2 in enumerate(other_cloud):
            p1_p    = p1[p.px:p.py+1]
            p1_mass = p1[p.mass]
            p2_p    = p2[p.px:p.py+1]
            p2_mass = p2[p.mass]
            dif = p1_p - p2_p
            dist = np.sqrt(np.sum(np.square(dif)))
            if dist == 0:
                continue
            f = (G * p1_mass * p2_mass) / (dist*dist)

            self_cloud[i1][p.ax] -= f * dif[0] / p1_mass
            self_cloud[i1][p.ay] -= f * dif[1] / p1_mass

            if update_other > 0:
                other_cloud[i2][p.ax] -= f * dif[0] / p1_mass
                other_cloud[i2][p.ay] -= f * dif[1] / p1_mass
