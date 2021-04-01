import numpy as np
import logging
import threading
from .config import Config
import barneshut.internals.particle as p
from barneshut.kernels.gravity import get_gravity_kernel
from numba import njit

class Cloud:
    
    def __init__(self, pre_alloc=None):
        self.max_particles = int(Config.get("grid", "max_particles_per_box"))
        self.COM = None
        self.n = 0        
        if pre_alloc is not None:
            self.__particles = np.empty((pre_alloc+1,p.nfields), dtype=p.ftype)
        else:
            self.__particles = None
        self.lock = threading.Lock()
        self.G = float(Config.get("bh", "grav_constant"))
        # Set our kernels
        self.grav_kernel = get_gravity_kernel()

    @staticmethod
    def from_slice(pslice):
        c = Cloud()
        c.add_particle_slice(pslice)
        return c

    #
    # general getter/setters
    @property
    def particles(self):
        return self.__particles[:self.n]

    @property
    def positions(self):
        return self.__particles[:self.n, p.px:p.py+1]
        
    @positions.setter
    def positions(self, pos):
        self.__particles[:self.n, p.px:p.py+1] = pos

    @property
    def velocities(self):
        return self.__particles[:self.n, p.vx:p.vy+1]

    @positions.setter
    def velocities(self, v):
        self.__particles[:self.n, p.vx:p.vy+1] = v

    @property
    def masses(self):
        return self.__particles[:self.n, p.mass:p.mass+1]

    @property
    def accelerations(self):
        return self.__particles[:self.n, p.ax:p.ay+1]

    @accelerations.setter
    def accelerations(self, acc):
        self.__particles[:self.n, p.ax:p.ay+1] = acc

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.max_particles

    def add_particles(self, ps):
        for p in ps:
            self.add_particle(p)

    def add_particle(self, p):
        if self.__particles is None:
            self.__particles = np.empty((self.max_particles+1,p.nfields), dtype=p.ftype)
        self.__particles[self.n] = p
        self.n += 1

    def add_particle_slice(self, pslice):
        self.__particles = pslice
        self.n = len(pslice)

    def get_COM(self):
        if self.COM is None:
            self.COM = Cloud(pre_alloc=1)
            # if we have no particle, COM is all zeros
            if self.is_empty():
                data = (0,) * p.nfields
            else:
                # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
                M = np.sum(self.masses)
                coords = self.positions * self.masses
                coords = np.add.reduce(coords)
                coords /= M
                data = [coords[0], coords[1], M] + [.0] * (p.nfields-3)
            point = np.array(data, dtype=p.ftype)
            self.COM.add_particle(point)

        return self.COM

    @njit(fastmath=True)
    def tick_particles(self):
        if self.n == 0:
            return
        tick = float(Config.get("bh", "tick_seconds"))
        self.velocities += self.accelerations * tick
        self.accelerations = 0.0               
        #logging.debug(f"changing position of particle from {self.positions[:3]}\nto {self.positions[:3]+self.velocities[:3] * tick}")
        self.positions += self.velocities * tick
    
    def apply_force(self, other_cloud, update_other):
        self.grav_kernel(self, other_cloud, self.G, update_other)