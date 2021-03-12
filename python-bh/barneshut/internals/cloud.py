import numpy as np
import logging
from numpy.lib.recfunctions import structured_to_unstructured as unst
from .particle import particle_type
from .config import Config
from barneshut.kernels.gravity import get_gravity_kernel


class Cloud:
    
    def __init__(self, pre_alloc=None):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.COM = None
        self.G = float(Config.get("bh", "grav_constant"))
        self.n = 0        
        if pre_alloc:
            self.__particles = np.empty((pre_alloc+1,), dtype=particle_type)
        else:
            self.__particles = None
        # Set our kernels
        self.__apply_force = get_gravity_kernel()

    #
    # Factories
    #
    @staticmethod
    def concatenation(cloud1, cloud2):
        c = Cloud(pre_alloc=cloud1.n + cloud2.n)
        for p in cloud1.particles:
            c.add_particle(p)   
        for p in cloud2.particles:
            c.add_particle(p)  
        return c
        
    #
    # general getter/setters
    #
    @property
    def particles(self):
        return self.__particles[:self.n]

    @property
    def positions(self):
        return unst(self.__particles, copy=False)[:self.n,:2]
        
    @positions.setter
    def positions(self, pos):
        unst(self.__particles, copy=False)[:self.n,:2] = pos

    @property
    def velocities(self):
        return unst(self.__particles, copy=False)[:self.n,3:5]

    @positions.setter
    def velocities(self, v):
        unst(self.__particles, copy=False)[:self.n,3:5] = v
        
    @property
    def masses(self):
        return unst(self.__particles, copy=False)[:self.n,2:3]
    
    @property
    def accelerations(self):
        return unst(self.__particles, copy=False)[:self.n,5:7]

    @accelerations.setter
    def accelerations(self, acc):
        unst(self.__particles, copy=False)[:self.n,5:7] = acc

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.max_particles

    def add_particles(self, ps):
        for p in ps:
            self.add_particle(p)

    def add_particle(self, p):
        # TODO: maybe add a decorator for this check
        if self.__particles is None:
            self.__particles = np.empty((self.max_particles+1,), dtype=particle_type)

        self.__particles[self.n] = p
        self.n += 1

    def add_particle_slice(self, pslice):
        self.__particles = pslice
        self.n = len(pslice)

    def get_COM(self):
        # TODO: need a switch here to use different COM kernels
        if self.COM is None:
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            M = np.sum(self.masses)
            coords = np.multiply(self.positions, self.masses)
            coords = np.add.reduce(coords)
            coords /= M
            self.COM = Cloud(pre_alloc=1)
            data = (coords[0], coords[1], M, .0, .0, .0, .0, .0, .0)
            p = np.array(data, dtype=particle_type)
            self.COM.add_particle(p)

        return self.COM

    def tick_particles(self):
        tick = float(Config.get("bh", "tick_seconds"))
        self.velocities += self.accelerations * tick
        self.accelerations[:,:] = 0.0               
        self.positions += self.velocities * tick

    def apply_force(self, other_cloud):
        self.__apply_force(self, other_cloud, self.G)
    