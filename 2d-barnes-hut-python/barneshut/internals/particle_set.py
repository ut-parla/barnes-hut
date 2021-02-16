from . import Particle
from .config import Config
import numpy as np
from scipy.spatial.distance import pdist, squareform

POS_X = 0
POS_Y = 1
MASS  = 3
VEL_X = 4
VEL_Y = 5
ACC_X = 6
ACC_Y = 7


class ParticleSet:

    def __init__(self):
        self.particles_per_node = int(Config.get("quadtree", "particles_per_leaf"))
        self.particles = np.ndarray((self.particles_per_node, 7))
        self.n = 0
        self.COM = None

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.particles_per_node

    def get_particles(self):
        return self.particles

    def add_particle(self, part):
        self.particles[self.n] = part
        self.n += 1

    def get_COM(self):
        if self.COM is None:
            #ppm = np.multiply.reduce(self.particles[:MASS], axis=1)
            
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            mx = np.multiply(self.particles[:,POS_X:POS_X+1], self.particles[:,MASS:MASS+1])
            my = np.multiply(self.particles[:,POS_Y:POS_Y+1], self.particles[:,MASS:MASS+1])
            M  = np.sum(self.particles[MASS:MASS+1])

            newX = np.divide(np.sum(mx), M)
            newY = np.divide(np.sum(my), M)

            self.COM = np.ndarray(7)
            self.COM[:3] = [newX, newY, M]

        return self.COM

    def apply_force(self, other_set, is_COM=False):
        # TODO: there has to be a way to do this using gemm or whatever, for now i just want to make it work
        # squareform(pdist(self.particles, other_set.particles))
        GRAV = Config.get("bh", "grav_constant")
        if not is_COM:
            for p1 in self.particles:
                for p2 in other_set.particles:
                    diff = p1[:POS_Y+1] - p2[:POS_Y+1]
                    dist = np.linalg.norm(diff)
                    f = (GRAV * p1[MASS:MASS+1] * p2[MASS:MASS+1]) / (dist*dist)

                    p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]
                    p2[ACC_X:ACC_Y+1] += (f * diff) / p2[MASS:MASS+1]

        else:
            com = other_set.get_COM()
            for p1 in self.particles:
                diff = p1[:POS_Y+1] - com[:POS_Y+1]
                dist = np.linalg.norm(diff)
                f = (GRAV * p1[MASS:MASS+1] * com[MASS:MASS+1]) / (dist*dist)

                p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]

            
    def tick_particles(self):
         # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf
        tick = float(Config.get("bh", "tick_seconds"))
     
        #self.position += self.velocity * tick/2
        # the second part probably causes a copy. we can do that in place, but we would have to undo before the next step
        self.particles[POS_X:POS_X+1] += (self.particles[VEL_X:VEL_X+1] * (tick/2))
        self.particles[POS_Y:POS_Y+1] += (self.particles[VEL_Y:VEL_Y+1] * (tick/2))
        
        #self.velocity += self.acceleration * tick
        self.particles[VEL_X:VEL_X+1] += (self.particles[ACC_X:ACC_X+1] * tick)
        self.particles[VEL_Y:VEL_Y+1] += (self.particles[ACC_Y:ACC_Y+1] * tick)

        #self.position += self.velocity * tick/2
        self.particles[POS_X:POS_X+1] += (self.particles[VEL_X:VEL_X+1] * (tick/2))
        self.particles[POS_Y:POS_Y+1] += (self.particles[VEL_Y:VEL_Y+1] * (tick/2))

        #self.acceleration = np.zeros(2)
        self.particles[ACC_X:ACC_Y+1] = .0



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
    #     tick = float(Config.get("bh", "tick_seconds"))
    #     self.position += self.velocity * tick/2
    #     self.velocity += self.acceleration * tick
    #     self.position += self.velocity * tick/2
    #     self.acceleration = np.zeros(2)
    
    # def combine_COM(self, otherCOM):
    #     added_mass = self.mass + otherCOM.mass
    #     self.position = ((self.position * self.mass) + (otherCOM.position * otherCOM.mass)) / added_mass
    #     self.mass = added_mass

    # def __repr__(self):
    #     return '<Particle x: {}, y:{}>'.format(*self.position)
