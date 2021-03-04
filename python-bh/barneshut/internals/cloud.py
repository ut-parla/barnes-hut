from . import Particle
from .config import Config
import numpy as np
from barneshut.gravkernels import get_kernel

# hopefully it won't be too hard to switch from 2d to 3d
N_DIM = 2

class Cloud:

    def __init__(self, com_particle=None, concatenation=None):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.COM = None

        # If we wanna get a cloud representation of a COM
        if com_particle is not None:
            self.n = 1
            # we allocate an extra element because of guvectorize cuda stuff
            self.__positions = np.array([com_particle.position, (0,0)])
            self.__masses = np.array([com_particle.mass,0.0])
            self.__velocities = np.ndarray((2, N_DIM))
            self.__accelerations = np.ndarray((2, N_DIM))
        # want to concatenate clouds
        elif concatenation is not None:
            c1, c2 = concatenation
            self.n = c1.n + c2.n
            self.__positions = np.concatenate((c1.positions, c2.positions))
            self.__masses = np.concatenate((c1.masses, c2.masses))
            self.__velocities = np.ndarray((self.n, N_DIM))
            self.__accelerations = np.ndarray((self.n, N_DIM))
        # if this is an empty Cloud creation
        else:
            self.n = 0
            # add one to max because of guvectorize cuda stuff
            self.__positions = np.ndarray((self.max_particles+1, N_DIM))
            self.__masses = np.ndarray(self.max_particles+1)
            self.__velocities = np.ndarray((self.max_particles+1, N_DIM))
            self.__accelerations = np.ndarray((self.max_particles+1, N_DIM))

        self.__apply_force = get_kernel()

    #
    # general getter/setters
    #
    @property
    def positions(self):
        return self.__positions[:self.n]

    @positions.setter
    def positions(self, pos):
        self.__positions[:self.n] = pos

    @property
    def velocities(self):
        return self.__velocities[:self.n]

    @positions.setter
    def velocities(self, v):
        self.__velocities[:self.n] = v

    @property
    def masses(self):
        return self.__masses[:self.n]
    
    @property
    def accelerations(self):
        return self.__accelerations[:self.n]

    @accelerations.setter
    def accelerations(self, acc):
        self.__accelerations[:self.n] = acc

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.max_particles

    def add_particle(self, part: Particle):
        self.__positions[self.n] = part.position
        self.__velocities[self.n] = part.velocity
        self.__masses[self.n] = part.mass
        self.__accelerations[self.n] = 0.0
        self.n += 1

    def concatenation(self, other_cloud):
        return Cloud(concatenation=(self, other_cloud))

    def get_COM(self):
        if self.COM is None:
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            M = np.sum(self.masses)
            coords = np.multiply(self.positions, self.masses[:, np.newaxis])
            coords = np.add.reduce(coords)
            coords /= M
            self.COM = Cloud(com_particle=Particle(coords, M))
        return self.COM

    def tick_particles(self):
        tick = float(Config.get("bh", "tick_seconds"))
        self.velocities += self.accelerations * tick
        self.accelerations[:,:] = 0.0               
        self.positions += self.velocities * tick

    def apply_force(self, other_cloud, use_COM=False):
        #if use_COM then we don't do all p2p computation, instead we get the COM of the cloud
        other = other_cloud if use_COM is False else other_cloud.get_COM()
        # pass use_COM in case the calculation needs to know if other is a COM or a set of particles
        self.__apply_force(other, use_COM)
    