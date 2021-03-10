import os

#os.environ['MKL_VERBOSE']="1"
os.environ['OMP_NUM_THREADS']="1"

import numpy as np
import itertools
from timeit import timeit
from scipy.linalg.blas import zhpr, dspr2, zhpmv, cgeru, dgemv, cgemv, dger
from numba import njit, jit, prange, guvectorize

from scipy.spatial.distance import cdist

@guvectorize(["float64[:], float64, float64[:,:], float64[:], float64, float64[:], float64[:,:]"],'(d),(), (n,d), (n), () -> (d), (n,d)', nopython=True, target="cpu", fastmath=True)
def internal_gu(p_pos, p_mass, cloud_positions, cloud_masses, G, p_accel, cloud_accels):
    n = cloud_positions.shape[0]
    for i in range(n):
        dif = p_pos-cloud_positions[i]
        dist = np.sqrt(np.sum(np.square(dif)))
        if dist > 0:
            f = (G * p_mass * cloud_masses[i]) / (dist*dist*dist)
        else:
            f = 0
        p_accel         -= (f * dif / p_mass)
        cloud_accels[i] += (f * dif / cloud_masses[i])

def self_self_gu(posA, masses):
    G = 1.0
    p_accel = np.zeros( (posA.shape[0], posA.shape[1]) )
    cloud_accel = np.zeros( (posA.shape[0], posA.shape[1]) )
    return internal_gu(posA, masses, posA, masses, G, p_accel, cloud_accel)

def self_other_gu(posA, posB, masses):
    G = 1.0
    p_accel = np.zeros( (posA.shape[0], posA.shape[1]) )
    cloud_accel = np.zeros( (posA.shape[0], posA.shape[1]) )
    return internal_gu(posA, masses, posB, masses, G, p_accel, cloud_accel)

@njit("float64[:, :](float64[:, :], float64[:],)", fastmath=True)
def self_self_numba(x, masses):
    G = 1.0
    field = np.zeros_like(x)
    N = x.shape[0]
    dx = np.empty(3)
    fk = 0.
    for i in range(N):
        for j in range(i+1,N):
            distSqr = 0.
            for k in range(3):
                dx[k] = x[j,k] - x[i,k]
                distSqr += dx[k]**2
            for k in range(3):
                fk = G*dx[k]*distSqr**-1.5
                field[i, k] += fk * masses[j]
                field[j, k] -= fk * masses[i]
    return field

def self_self_pm(positions, masses, G=1):
    mass_matrix = masses.reshape((1, -1, 1))*masses.reshape((-1, 1, 1))
    disps = positions.reshape((1, -1, 3)) - positions.reshape((-1, 1, 3)) # displacements
    dists = np.linalg.norm(disps, axis=2)
    dists[dists == 0] = 1 # Avoid divide by zero warnings
    forces = G*disps*mass_matrix/np.expand_dims(dists, 2)**3
    return forces.sum(axis=1)/masses.reshape(-1, 1)

def self_self_vec(pos, mas):
    n = mas.size
    d2 = pos@(-2*pos.T)
    diag = -0.5 * np.einsum('ii->i', d2)
    d2 += diag + diag[:, None]
    np.einsum('ii->i', d2)[...] = 1
    return np.nansum((pos[:, None, :] - pos) * (mas[:, None] * d2**-1.5)[..., None], axis=0)


def self_other_vec(posA, posB,mas):
    n = mas.size
    D = cdist(posA, posB)**2
    return np.nansum((posA[:, None] - posB[None, :]) * (mas[:, None] * D**-1.5)[..., None], axis=0)

@njit("float64[:, :](float64[:, :], float64[:, :], float64[:],)", fastmath=True)
def self_other_numba(x, y, masses):
    G = 1.0
    field = np.zeros_like(x)
    N = x.shape[0]
    dx = np.empty(3)
    fk = 0.
    for i in range(N):
        for j in range(N):
            distSqr = 0.
            for k in range(3):
                dx[k] = x[j,k] - y[i,k]
                distSqr += dx[k]**2
            for k in range(3):
                fk = G*dx[k]*distSqr**-1.5
                #field[i, k] += fk * masses[j]
                field[j, k] -= fk * masses[i]
    return field

def self_self_blas(pos, mas):
    n = mas.size
    trck = np.zeros((3, n * (n + 1) // 2), complex)
    for a, p in zip(trck, pos.T - 1j):
        zhpr(n, -2, p, a, 1, 0, 0, 1)

    ppT = trck.real.sum(0) + 6

    dspr2(n, -0.5, ppT[np.r_[0, 2:n+1].cumsum()], np.ones((n,)), ppT,
          1, 0, 1, 0, 0, 1)

    np.divide(trck.imag, ppT*np.sqrt(ppT), where=ppT.astype(bool),
              out=trck.imag)

    out = np.zeros((3, n), complex)
    for a, o in zip(trck, out):
        zhpmv(n, 0.5, a, mas*-1j, 1, 0, 0, o, 1, 0, 0, 1)

    return out.real.T


@njit("UniTuple(complex128[:, :], 2)(float64[:, :], float64[:, :])", fastmath=True)
def prepare_complex_numba(posA, posB):
    posA_mod = posA - 1j
    posB_mod = posB + 1j
    return posA_mod, posB_mod

def prepare_complex(posA, posB):
    posA_mod = posA - 1j
    posB_mod = posB + 1j
    return posA_mod, posB_mod

@njit("float64[:](float64[:, :])", fastmath=True)
def norm_internal_numba(X):
    out = np.empty(X.shape[0])
    for i in prange(X.shape[0]):
        out[i] = np.linalg.norm(X[i, :], ord=2)**2
    return out

#Note: This is slower than pure numpy apparently :/
@njit("UniTuple(float64[:], 2)(float64[:, :], float64[:, :])", fastmath=True)
def norm_numba(posA, posB):
    posA_norm = norm_internal_numba(posA)
    posB_norm = norm_internal_numba(posB)
    return posA_norm, posB_norm

def norm(posA, posB):
    posA_norm = np.linalg.norm(posA, ord=2, axis=1)**2
    posB_norm = np.linalg.norm(posB, ord=2, axis=1)**2
    return posA_norm, posB_norm

def self_other_blas(posA, posB, mas, posA_norm=None, posB_norm=None, ones=None):
    n = mas.size

    cstore = np.asarray(np.zeros((3, n, n)), dtype=np.complex128, order="F")
    posA_mod, posB_mod = prepare_complex_numba(posA, posB)

    for a, pa, pb in zip(cstore, posA_mod.T, posB_mod.T):
        a[:]= cgeru(-2.0, pa, pb, overwrite_x=0, overwrite_y=0, a=a, overwrite_a=0)

    D = cstore.real.sum(0) + 6

    if posA_norm is None:
        posA_norm = np.linalg.norm(posA, ord=2, axis=1)**2
    if posB_norm is None:
        posB_norm = np.linalg.norm(posB, ord=2, axis=1)**2
    if ones is None:
        ones = np.ones((n), dtype=np.float64, order="F")

    dger(1.0, posA_norm, ones, overwrite_x=0, overwrite_y=0, a=D, overwrite_a=1)
    dger(1.0, ones, posB_norm, overwrite_x=0, overwrite_y=0, a=D, overwrite_a=1)

    np.divide(cstore.imag, D*np.sqrt(D), where=D.astype(bool), out=cstore.imag)

    out = np.zeros((3, n), dtype=np.float64)
    for a, o in zip(cstore, out):
        o[:] = dgemv(0.5, a.imag, mas)
    
    return out.real

def multiplication_test(pos):
    return pos @ pos.T

N = 1024
np.random.seed(10)
A = np.asarray(np.random.randn(N, 3), order="F")
B = np.asarray(np.random.randn(N, 3), order="F")
mas = np.asarray(np.random.randn(N), order='F')
pos = A 
t =1

"""
#Check Correctness 
a = self_other_numba(A, B,mas)
print(a)
a = self_other_gu(A, B, mas)
print(a[0])
"""


print(f"test_self_blas:      {timeit('self_self_blas(A, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"test_self_numba:      {timeit('self_self_numba(A, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"test_self_guvectorize:      {timeit('self_self_gu(A, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"test_self_numpy:      {timeit('self_self_pm(A, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"test_self_numpy_2:      {timeit('self_self_vec(A, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"test_other_blas:      {timeit('self_other_blas(A, B, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"self_other_numba:      {timeit('self_other_numba(A, B, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")
print(f"self_other_gu:      {timeit('self_other_gu(A, B, mas)', globals=globals(), number=t)*1000/t:10.3f} ms")