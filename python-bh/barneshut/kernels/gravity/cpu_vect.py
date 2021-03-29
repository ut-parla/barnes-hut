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

#@njit("(float64[:, :], float64[:, :], float64, float64)", fastmath=True)
def cpu_vect_kernel(self_cloud, other_cloud, G, update_other):
    # get positions and masses of concatenation
    for i in range(self_cloud.shape[0]):
        for j in range(other_cloud.shape[0]):
            p1_p    = self_cloud[i, p.px:p.py+1]
            p1_mass = self_cloud[i, p.mass]
            p2_p    = other_cloud[j, p.px:p.py+1]
            p2_mass = other_cloud[j, p.mass]
            dif = p1_p - p2_p
            dist = np.sqrt(np.sum(np.square(dif)))
            if dist == .0:
                continue
            f = (G * p1_mass * p2_mass) / (dist*dist)

            self_cloud[i, p.ax] -= f * dif[0] / p1_mass
            self_cloud[i, p.ay] -= f * dif[1] / p1_mass

            if update_other > 0:
                other_cloud[j, p.ax] += f * dif[0] / p2_mass
                other_cloud[j, p.ay] += f * dif[1] / p2_mass

        #print(f"point {self_cloud[i, p.pid]} = {self_cloud[i, p.ax]}, {self_cloud[i, p.ay]}")
