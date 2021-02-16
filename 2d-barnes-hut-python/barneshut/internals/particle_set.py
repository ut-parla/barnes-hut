from . import Particle
from .config import Config
from numba import f4, i4
from numba.experimental import jitclass
import numpy as np

POS_X = 0
POS_Y = 1
VEL_X = 2
VEL_Y = 3
MASS  = 4
ACC_X = 5
ACC_Y = 6


class ParticleSet:

    def __init__(self):
        self.particles_per_node = Config.get("quadtree", "particles_per_leaf")
        self.particles = np.ndarray((self.particles_per_node, 7))
        self.n_particles = 0
        self.COM = None

    def is_empty(self):
        return self.n_particles == 0

    def is_full(self):
        return self.n_particles >= self.particles_per_node

    def get_particles(self):
        return self.particles

    def add_particle(self, part: Particle):
        self.particles[self.n_particles] = part.get_array()
        self.n_particles += 1

    def get_COM(self):
        if self.COM is None:
            pass
            # calculate COM

        return self.COM

    def apply_force(self, other_set, is_COM=False):
        pass

    def tick_particles(self):
        pass


    # # G = 6.673 x 10-11 Nm^2/kg^2
    # # Fgrav = (G*m1*m2)/d^2
    # # F = m*a
    # def apply_force(self, other, is_COM=False):
    #     diff = self.position - other.position
    #     dist = np.linalg.norm(diff)
    #     f = (Config.get("bh", "grav_constant") * self.mass * other.mass) / (dist*dist)

    #     # update self acceleration
    #     self.acceleration -= (f * diff) / self.mass
    #     # update other particles acceleration
    #     if not is_COM:
    #         other.acceleration += (f * diff) / self.mass

    # def tick(self):
    #     # this looks wrong and I dont know why. Cant find a reliable source for this equation
    #     # this is from https://www.cs.utexas.edu/~rossbach/cs380p/lab/bh-submission-cs380p.html
    #     #self.pos.x += (cn.TICK_SECONDS * self.velocity.x) + (0.5 * self.accel.x * cn.TICK_SECONDS*cn.TICK_SECONDS)
    #     #self.pos.y += (cn.TICK_SECONDS * self.velocity.y) + (0.5 * self.accel.y * cn.TICK_SECONDS*cn.TICK_SECONDS)

    #     # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf
    #     tickt = float(Config.get("bh", "tick_seconds"))
    #     self.position += self.velocity * tickt/2
    #     self.velocity += self.acceleration * tickt
    #     self.position += self.velocity * tickt/2
    #     self.acceleration = np.zeros(2)
    
    # def combine_COM(self, otherCOM):
    #     added_mass = self.mass + otherCOM.mass
    #     self.position = ((self.position * self.mass) + (otherCOM.position * otherCOM.mass)) / added_mass
    #     self.mass = added_mass

    # def __repr__(self):
    #     return '<Particle x: {}, y:{}>'.format(*self.position)
