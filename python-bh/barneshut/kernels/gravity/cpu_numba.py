import numpy as np
from barneshut.internals import Cloud
from numba import njit, jit, prange

def get_kernel_function():
    return cpu_numba_kernel

def cpu_numba_kernel(self_cloud, other_cloud, G, update_other=False):
    if self_cloud == other_cloud:
        posA = self_cloud.positions
        masA = self_cloud.masses.squeeze(axis=1)
        acc = self_self_numba(posA, masA, G)
        self_cloud.accelerations  += acc

    else:
        masA = self_cloud.masses.squeeze(axis=1)
        posA = self_cloud.positions

        masB = other_cloud.masses.squeeze(axis=1)
        posB = other_cloud.positions

        if update_other:
            acc1, acc2 = self_other_numba_2(posA, posB, masA, masB, G)
            self_cloud.accelerations += acc1
            other_cloud.accelerations  += acc2
        else:
            acc1  = self_other_numba_1(posA, posB, masB, G)
            self_cloud.accelerations += acc1

@njit("float64[:, :](float64[:, :], float64[:],float64,)", fastmath=True)
def self_self_numba(x, masses, G):
    field = np.zeros_like(x)
    N = x.shape[0]
    dx = np.empty(2)
    fk = 0.
    eps = 1e-5
    for i in range(N):
        for j in range(i+1,N):
            distSqr = 0.
            for k in range(2):
                dx[k] = x[j,k] - x[i,k]
                distSqr += dx[k]**2
            for k in range(2):
                fk = G*dx[k]*distSqr**-1 if distSqr > eps else 0
                field[i, k] += fk * masses[j]
                field[j, k] -= fk * masses[i]
    return field

@njit("UniTuple(float64[:, :], 2)(float64[:, :], float64[:, :], float64[:], float64[:], float64,)", fastmath=True)
def self_other_numba_2(x, y, masA, masB, G):
    field1 = np.zeros_like(x)
    field2 = np.zeros_like(y)

    Na = x.shape[0]
    Nb = y.shape[0]

    dx = np.empty(2)
    fk = 0.
    eps = 1e-5
    for i in range(Nb):
        for j in range(Na):
            distSqr = 0.
            for k in range(2):
                dx[k] = x[j,k] - y[i,k]
                distSqr += dx[k]**2
            for k in range(2):
                fk = G*dx[k]*distSqr**-1 if distSqr > eps else 0
                field1[j, k] -= fk * masB[i]
                field2[i, k] -= fk * masA[j]
    return field1, field2

@njit("float64[:, :](float64[:, :], float64[:, :], float64[:], float64,)", fastmath=True)
def self_other_numba_1(x, y, masB, G):
    field1 = np.zeros_like(x)

    Na = x.shape[0]
    Nb = y.shape[0]

    dx = np.empty(2)
    fk = 0.
    eps = 1e-5
    for i in range(Nb):
        for j in range(Na):
            distSqr = 0.
            for k in range(2):
                dx[k] = x[j,k] - y[i,k]
                distSqr += dx[k]**2
            for k in range(2):
                fk = G*dx[k]*distSqr**-1 if distSqr > eps else 0
                field1[j, k] -= fk * masB[i]

    return field1
