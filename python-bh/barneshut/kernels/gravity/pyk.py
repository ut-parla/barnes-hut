from math import sqrt

import numpy as np
import pykokkos as pk

from barneshut.internals.config import Config

def get_kernel_function():
    return pyk_run

@pk.functor(
    x=pk.ViewTypeInfo(trait=pk.Unmanaged),
    masses=pk.ViewTypeInfo(trait=pk.Unmanaged)
)
class GravitySelfSelfFunctor:
    def __init__(self, x, masses, G):
        x = np.array(x, copy=True)
        masses = np.array(masses, copy=True)

        self.x: pk.View2D[pk.double] = pk.from_numpy(x)
        self.masses: pk.View1D[pk.double] = pk.from_numpy(masses)
        self.field: pk.View2D[pk.double] = pk.View(self.x.shape, pk.double)

        self.N: int = self.x.extent(0)
        self.G: float = G
        self.eps: float = 1e-5

    @pk.workunit
    def run(self, i: int):
        for j in range(i + 1, self.N):
            dx_0: float = self.x[j][0] - self.x[i][0]
            dx_1: float = self.x[j][1] - self.x[i][1]

            distSqr: float = (dx_0 * dx_0) + (dx_1 * dx_1)

            fk_0: float = 0
            fk_1: float = 0
            if distSqr > self.eps:
                dist_reciprocal: float = 1 / distSqr
                fk_0 = self.G * dx_0 * dist_reciprocal
                fk_1 = self.G * dx_1 * dist_reciprocal

            self.field[i][0] += fk_0 * self.masses[j]
            self.field[j][0] -= fk_0 * self.masses[i]
            self.field[i][1] += fk_1 * self.masses[j]
            self.field[j][1] -= fk_1 * self.masses[i]

@pk.functor(
    p_pos=pk.ViewTypeInfo(trait=pk.Unmanaged),
    cloud_pos=pk.ViewTypeInfo(trait=pk.Unmanaged),
    cloud_mass=pk.ViewTypeInfo(trait=pk.Unmanaged)
)
class GravitySelfOtherFunctor:
    def __init__(self, c1_positions, c2_positions, c2_masses, G):
        c1_positions = np.array(c1_positions, copy=True)
        c2_positions = np.array(c2_positions, copy=True)
        c2_masses = np.array(c2_masses, copy=True)

        self.p_pos: pk.View2D[pk.double] = pk.from_numpy(c1_positions)
        self.p_accel: pk.View2D[pk.double] = pk.View(self.p_pos.shape, pk.double)
        self.cloud_pos: pk.View2D[pk.double] = pk.from_numpy(c2_positions)
        self.cloud_mass: pk.View1D[pk.double] = pk.from_numpy(c2_masses)

        self.n: int = self.cloud_pos.extent(0)
        self.G: float = G
        self.eps: float = 1e-5

    @pk.workunit
    def run(self, tid: int):
        for i in range(self.n):
            dif_1: float = self.p_pos[tid][0] - self.cloud_pos[i][0]
            dif_2: float = self.p_pos[tid][1] - self.cloud_pos[i][1]

            dist: float = (dif_1 * dif_1) + (dif_2 * dif_2)
            f_1: float = 0
            f_2: float = 0
            if dist > self.eps:
                dist_reciprocal: float = 1 / dist
                f_1 = self.G * dif_1 * dist_reciprocal
                f_2 = self.G * dif_2 * dist_reciprocal

            self.p_accel[tid][0] -= f_1 * self.cloud_mass[i]
            self.p_accel[tid][1] -= f_2 * self.cloud_mass[i]


def pyk_run(self_cloud, other_cloud, G, update_other):
    if self_cloud == other_cloud:
        posA = self_cloud.positions
        masA = self_cloud.masses.squeeze(axis=1)
    
        f = GravitySelfSelfFunctor(posA, masA, G)
        N = self_cloud.positions.shape[0]
        pk.parallel_for(N, f.run)

        self_cloud.accelerations += f.field

    else:
        masA = self_cloud.masses.squeeze(axis=1)
        posA = self_cloud.positions

        masB = other_cloud.masses.squeeze(axis=1)
        posB = other_cloud.positions

        if update_other:
            pass

        else:
            f = GravitySelfOtherFunctor(posA, posB, masB, G)
            N = self_cloud.positions.shape[0]
            pk.parallel_for(N, f.run)

            self_cloud.accelerations += f.p_accel.data
