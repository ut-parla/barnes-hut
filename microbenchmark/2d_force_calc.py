import numpy as np
import itertools
from scipy.linalg.blas import zhpr, dspr2, zhpmv

#
#
#  This code was taken from https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python
#  Modified to 2d for testing
#

def acc_vect(pos, mas):
    n = mas.size
    d2 = pos@(-2*pos.T)
    diag = -0.5 * np.einsum('ii->i', d2)
    d2 += diag + diag[:, None]
    np.einsum('ii->i', d2)[...] = 1
    return np.nansum((pos[:, None, :] - pos) * (mas[:, None] * d2**-1.5)[..., None], axis=0)

def acc_blas(pos, mas):
    n = mas.size
    # trick: use complex Hermitian to get the packed anti-symmetric
    # outer difference in the imaginary part of the zhpr answer
    # don't want to sum over dimensions yet, therefore must do them one-by-one
    trck = np.zeros((3, n * (n + 1) // 2), complex)
    for a, p in zip(trck, pos.T - 1j):
        zhpr(n, -2, p, a, 1, 0, 0, 1)
        # does  a  ->  a + alpha x x^H
        # parameters: n             --  matrix dimension
        #             alpha         --  real scalar
        #             x             --  complex vector
        #             ap            --  packed Hermitian n x n matrix a
        #                               i.e. an n(n+1)/2 vector
        #             incx          --  x stride
        #             offx          --  x offset
        #             lower         --  is storage of ap lower or upper
        #             overwrite_ap  --  whether to change a inplace
    # as a by-product we get pos pos^T:
    ppT = trck.real.sum(0) + 6
    # now compute matrix of squared distances ...
    # ... using (A-B)^2 = A^2 + B^2 - 2AB
    # ... that and the outer sum X (+) X.T equals X ones^T + ones X^T
    dspr2(n, -0.5, ppT[np.r_[0, 2:n+1].cumsum()], np.ones((n,)), ppT,
          1, 0, 1, 0, 0, 1)
    # does  a  ->  a + alpha x y^T + alpha y x^T    in packed symmetric storage
    # scale anti-symmetric differences by distance^-3
    np.divide(trck.imag, ppT*np.sqrt(ppT), where=ppT.astype(bool),
              out=trck.imag)
    # it remains to scale by mass and sum
    # this can be done by matrix multiplication with the vector of masses ...
    # ... unfortunately because we need anti-symmetry we need to work
    # with Hermitian storage, i.e. complex numbers, even though the actual
    # computation is only real:
    out = np.zeros((2, n), complex)
    for a, o in zip(trck, out):
        zhpmv(n, 0.5, a, mas*-1j, 1, 0, 0, o, 1, 0, 0, 1)
        # multiplies packed Hermitian matrix by vector
    return out.real.T

def accelerations(positions, masses, epsilon=1e-6, gravitational_constant=1.0):
    '''Params:
    - positions: numpy array of size (n,3)
    - masses: numpy array of size (n,)
    '''
    n_bodies = len(masses)
    accelerations = np.zeros([n_bodies,2]) # n_bodies * (x,y,z)

    # vectors from mass(i) to mass(j)
    D = np.zeros([n_bodies,n_bodies,2]) # n_bodies * n_bodies * (x,y,z)
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        D[i][j] = positions[j]-positions[i]

    # Acceleration due to gravitational force between each pair of bodies
    A = np.zeros((n_bodies, n_bodies,2))
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        if np.linalg.norm(D[i][j]) > epsilon:
            A[i][j] = gravitational_constant * masses[j] * D[i][j] \
            / np.linalg.norm(D[i][j])**3

    # Calculate net accleration of each body
    accelerations = np.sum(A, axis=1) # sum of accel vectors for each body

    return accelerations

from numpy.linalg import norm

def acc_pm(positions, masses, G=1):
    '''Params:
    - positions: numpy array of size (n,3)
    - masses: numpy array of size (n,)
    '''
    mass_matrix = masses.reshape((1, -1, 1))*masses.reshape((-1, 1, 1))
    disps = positions.reshape((1, -1, 2)) - positions.reshape((-1, 1, 2)) # displacements
    dists = norm(disps, axis=2)
    dists[dists == 0] = 1 # Avoid divide by zero warnings
    forces = G*disps*mass_matrix/np.expand_dims(dists, 2)**3
    return forces.sum(axis=1)/masses.reshape(-1, 1)

n = 500
pos = np.random.random((n, 2))
mas = np.random.random((n,))

from timeit import timeit

print(f"loops:      {timeit('accelerations(pos, mas)', globals=globals(), number=1)*1000:10.3f} ms")
print(f"pmende:     {timeit('acc_pm(pos, mas)', globals=globals(), number=10)*100:10.3f} ms")
print(f"vectorized: {timeit('acc_vect(pos, mas)', globals=globals(), number=10)*100:10.3f} ms")
print(f"blas:       {timeit('acc_blas(pos, mas)', globals=globals(), number=10)*100:10.3f} ms")

A = accelerations(pos, mas)
AV = acc_vect(pos, mas)
AB = acc_blas(pos, mas)
AP = acc_pm(pos, mas)

#print(f"A: {A[:5]}")
#print(f"AB: {AB[:5]}")
#print(f"AV: {AV[:5]}")

assert np.allclose(A, AV) and np.allclose(AB, AV) and np.allclose(AP, AV)
