import numpy as np
from scipy.linalg.blas import zhpr, dspr2, zhpmv, cgeru, dgemv, cgemv, dger
from barneshut.internals import Cloud
from numba import njit, jit, prange

# calculations below are from this source:
# https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python

# this is what is called from __init__.py
def get_kernel_function():
    return cpu_blas_kernel

def cpu_blas_kernel(self_cloud, other_cloud, G, update_other=False):
    if self_cloud == other_cloud:
        blas_self_self(self_cloud, self_cloud, G)
    else:
        blas_self_other(self_cloud, other_cloud, G, update_other)


@njit("UniTuple(complex128[:, :], 2)(float64[:, :], float64[:, :])", fastmath=True)
def prepare_complex_numba(posA, posB):
    posA_mod = posA - 1j
    posB_mod = posB + 1j
    return posA_mod, posB_mod

def norm(posA, posB):
    posA_norm = np.linalg.norm(posA, ord=2, axis=1)**2
    posB_norm = np.linalg.norm(posB, ord=2, axis=1)**2
    return posA_norm, posB_norm

def blas_self_self(self_cloud, self_cloud2, G):
    mas = self_cloud.masses 
    pos = self_cloud.positions 

    n = mas.size 
    trck = np.zeros((2, n * (n + 1) // 2), complex)
    for a, p in zip(trck, pos.T - 1j):
        zhpr(n, -2, p, a, 1, 0, 0, 1)

    ppT = trck.real.sum(0) + 3

    dspr2(n, -0.5, ppT[np.r_[0, 2:n+1].cumsum()], np.ones((n,)), ppT,
        1, 0, 1, 0, 0, 1)

    np.divide(trck.imag, ppT, where=ppT.astype(bool),
            out=trck.imag)

    out = np.zeros((2, n), complex)
    for a, o in zip(trck, out):
        zhpmv(n, 0.5, a, mas*-1j, 1, 0, 0, o, 1, 0, 0, 1)
    acc = out.real.T

    # add accelerations
    self_cloud.accelerations  += acc

def blas_self_other(self_cloud, other_cloud, G, update_other=False):
    masA = self_cloud.masses 
    posA = self_cloud.positions
    nA = len(masA)

    masB = other_cloud.masses
    posB = other_cloud.positions
    nB = len(masB)

    cstore = np.asarray(np.zeros((2, nA, nB)), dtype=np.complex128, order='F')
    posA_mod, posB_mod = prepare_complex_numba(posA, posB)

    for a, pa, pb in zip(cstore, posA_mod.T, posB_mod.T):
            a[:]= cgeru(-2.0, pa, pb, overwrite_x=0, overwrite_y=0, a=a, overwrite_a=0)

    D = cstore.real.sum(0) + 4

    posA_norm = np.linalg.norm(posA, ord=2, axis=1)**2
    posB_norm = np.linalg.norm(posB, ord=2, axis=1)**2

    ones = np.ones((nB), dtype=np.float64, order="F")
    dger(1.0, posA_norm, ones, overwrite_x=0, overwrite_y=0, a=D, overwrite_a=1)

    ones = np.ones((nA), dtype=np.float64, order="F")
    dger(1.0, ones, posB_norm, overwrite_x=0, overwrite_y=0, a=D, overwrite_a=1)

    np.divide(cstore.imag, D, where=D.astype(bool), out=cstore.imag)

    #Compute incoming accelerations
    out = np.zeros((2, nA), dtype=np.float64)
    for a, o in zip(cstore, out):
        o[:] = dgemv(0.5, a.imag, masB)

    acc = out.T 
    self_cloud.accelerations  += acc

    if update_other:
        #Compute outgoing accelerations
        out = np.zeros((2, nB), dtype=np.float64)
        for a, o in zip(cstore, out):
            o[:] = dgemv(-0.5, a.imag.T, masA)
        acc = out.T 
        other_cloud.accelerations  += acc

#Alternative rank2-update kernel
@njit(fastmath=True)
def rank2_update(A, v, w):
    n = len(v)
    m = len(w)
    for i in range(n):
        for j in range(m):
            A[i,j] += v[i] + w[j]
    return A

#TODO: Add a better numpy self-other version here. The BLAS is slow due to memory and other problems 