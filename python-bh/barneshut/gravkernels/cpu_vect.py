import numpy as np
from numpy.linalg import norm

# calculations below are from this source:
# https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python

# this is what is called from __init__.py
def get_kernel_function():
    return cpu_vect_kernel

def cpu_vect_kernel(self_cloud, other_cloud, G, is_COM):
    # get positions and masses of concatenation
    cc = self_cloud.concatenation(other_cloud)
    masses = cc.masses
    positions = cc.positions

    # actual calculation
    mass_matrix = masses.reshape((1, -1, 1))*masses.reshape((-1, 1, 1))
    disps = positions.reshape((1, -1, 2)) - positions.reshape((-1, 1, 2)) # displacements
    dists = norm(disps, axis=2)
    dists[dists == 0] = 1 # Avoid divide by zero warnings
    forces = G*disps*mass_matrix/np.expand_dims(dists, 2)**3
    acc = forces.sum(axis=1)/masses.reshape(-1, 1)

    # add accelerations
    self_cloud.accelerations  += acc[:self_cloud.n,:]
    other_cloud.accelerations += acc[self_cloud.n:,:]
