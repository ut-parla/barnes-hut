import numpy as np
import logging
from .config import Config

import barneshut.internals.particle as p

class Cloud:
    
    def __init__(self, grav_kernel, pre_alloc=None):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.COM = None
        self.n = 0        
        if pre_alloc is not None:
            self.__particles = np.empty((pre_alloc+1,p.nfields), dtype=p.ftype)
        else:
            self.__particles = None
        
        self.G = float(Config.get("bh", "grav_constant"))
        # Set our kernels
        self.grav_kernel = grav_kernel

    #
    # Factories
    #
    @staticmethod
    def concatenation(cloud1, cloud2):
        c = Cloud(self.grav_kernel, pre_alloc=cloud1.n + cloud2.n)
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
        # TODO: maybe add a decorator for this check
        if self.__particles is None:
            self.__particles = np.empty((self.max_particles+1,p.nfields), dtype=p.ftype)

        self.__particles[self.n] = p
        self.n += 1

    def add_particle_slice(self, pslice):
        self.__particles = pslice
        self.n = len(pslice)

    def get_COM(self):
        # TODO: need a switch here to use different COM kernels
        if self.COM is None:
            self.COM = Cloud(self.grav_kernel, pre_alloc=1)
            # if we have no particle, COM is all zeros
            if self.is_empty():
                data = (0,) * p.nfields
            else:
                # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
                M = np.sum(self.masses)
                #coords = np.multiply(self.positions, self.masses)
                coords = self.positions * self.masses

                coords = np.add.reduce(coords)
                coords /= M
                data = (coords[0], coords[1], M, .0, .0, .0, .0, .0, .0)
            point = np.array(data, dtype=p.ftype)
            self.COM.add_particle(point)

        return self.COM

    def tick_particles(self):
        if self.n == 0:
            return
        tick = float(Config.get("bh", "tick_seconds"))
        self.velocities += self.accelerations * tick
        self.accelerations[:,:] = 0.0               
        self.positions += self.velocities * tick

    def apply_force(self, other_cloud, update_other=False):
        self.grav_kernel(self, other_cloud, self.G, update_other)
    