import numpy as np
from numba import guvectorize, float64, int64, njit, cuda, jit
from math import sqrt


# this is what is called from __init__.py
def get_kernel_function():
    return guvect_cuda


def guvect_cuda(self_cloud, other_cloud, G):
        # let's do the biggest set as first parameter, since guvectorize parallelize based on it's shape
        if self_cloud.n >= other_cloud.n:
            c1, c2 = self_cloud, other_cloud
        else:
            c1, c2 = other_cloud, self_cloud
        #print(f"c1 has {c1.n} particles, c2 has {c2.n}")

        # hack to change the dimension of output array, since v2 puts the c1 particle accelerations
        # at the last element of each output matrix.
        # this is why there's this whole slicing magic afterwards
        c2.n += 1
        is_self_self = 1.0 if c1 is c2 else 0.0
        accs = guvect_point_to_cloud_cuda(c1.positions, c1.masses, c2.positions, c2.masses, G, is_self_self)
        c2.n -= 1

        c2_sliced = accs[:,:-1]
        c2_acc = np.add.reduce(c2_sliced)        
        #print(f"c2_acc\n{c2_acc}")

        n = c2.n
        #print(f"accs\n{accs}")
        c1_sliced = accs[:, n::n]
        #print(f"c1_sliced:\n{c1_sliced}\n")
        c1_acc = c1_sliced.squeeze(axis=1)
        #print(f"c1_acc:\n{c1_acc}\n")
        #input("Press Enter to continue...")

        if is_self_self:
            c1.accelerations += c1_acc[:-1]
        else:
            c1.accelerations += c1_acc
        c2.accelerations += np.add.reduce(c2_acc)


# beware, cuda 11.1.x is the latest supported: https://github.com/numba/numba/issues/6607
@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64, float64[:,:]"], 
             '(d),(), (n,d), (n), (), () -> (n,d)', nopython=True, target="cuda")
def guvect_point_to_cloud_cuda(p_pos, p_mass, cloud_positions, cloud_masses, G, is_self_self, cloud_accels):
    # not sure if necessary, there is no documentation on initializing output
    cloud_accels[:, :] = 0.0
    n = cloud_positions.shape[0]-1
    p_i = n

    for i in range(n):
        x, y = p_pos[0] - cloud_positions[i][0], p_pos[1] - cloud_positions[i][1]
        dist = sqrt((x*x) + (y*y))
        f = (G * p_mass * cloud_masses[i]) / (dist*dist)
        cloud_accels[p_i][0] -= (f * x / p_mass)
        cloud_accels[p_i][1] -= (f * y / p_mass)
        
        if is_self_self != 0:
            cloud_accels[i][0] += (f * x / cloud_masses[i])
            cloud_accels[i][1] += (f * y / cloud_masses[i])
