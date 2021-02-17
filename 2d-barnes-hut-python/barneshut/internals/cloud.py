from . import Particle
from .config import Config
import numpy as np
from scipy.spatial.distance import pdist, squareform

POS_X = 0
POS_Y = 1
MASS  = 2
VEL_X = 3
VEL_Y = 4
ACC_X = 5
ACC_Y = 6


np.seterr(all='raise')

class Cloud:

    def __init__(self):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.particle_array = np.ndarray((self.max_particles, 7))
        self.n = 0
        self.COM = None

    @property
    def particles(self):
        return self.particle_array[:self.n]

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        #print(f"is_full  n:{self.n}  max:{self.max_particles} ")
        return self.n >= self.max_particles

    def add_particle(self, part):
        self.particle_array[self.n] = part
        self.n += 1

    def get_COM(self):
        if self.COM is None:
            #ppm = np.multiply.reduce(self.particles[:MASS], axis=1)
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            mx = np.multiply(self.particles[:,POS_X:POS_X+1], self.particles[:,MASS:MASS+1])
            my = np.multiply(self.particles[:,POS_Y:POS_Y+1], self.particles[:,MASS:MASS+1])
            M  = np.sum(self.particles[:, MASS:MASS+1])

            #print(f"COM: {mx}\n{my}\n{M}")

            newX = np.divide(np.sum(mx), M)
            newY = np.divide(np.sum(my), M)

            self.COM = np.zeros(7)
            self.COM[:3] = [newX, newY, M]

        return self.COM

    def apply_force(self, other_set, is_COM=False):
        # TODO: there has to be a way to do this using gemm or whatever, for now i just want to make it work
        # squareform(pdist(self.particles, other_set.particles))
        GRAV = float(Config.get("bh", "grav_constant"))
        if not is_COM:
            for p1 in self.particles:
                for p2 in other_set.particles:
                    try:
                        diff = p1[:POS_Y+1] - p2[:POS_Y+1]
                        dist = np.linalg.norm(diff)
                        f = (GRAV * p1[MASS:MASS+1] * p2[MASS:MASS+1]) / (dist*dist)

                        p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]
                        p2[ACC_X:ACC_Y+1] += (f * diff) / p2[MASS:MASS+1]
                    except Exception as e:
                        print(f"FORCE: diff: {diff}\ndist: {dist}\nf: {f}\nM1: {p1[MASS:MASS+1]}\nM2: {p2[MASS:MASS+1]}")
                        raise e
        else:
            com = other_set.get_COM()
            for p1 in self.particles:
                diff = p1[:POS_Y+1] - com[:POS_Y+1]
                dist = np.linalg.norm(diff)
                f = np.divide(GRAV * p1[MASS:MASS+1] * com[MASS:MASS+1], dist*dist)

                #print(f"FORCE: diff: {diff}\ndist: {dist}\nf: {f}\nACC: {p1[ACC_X:ACC_Y+1]}")

                p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]

            
    def tick_particles(self):
         # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf
        #print(f"TICK: POS: {self.particles[POS_X:POS_X+1]}\nVEL: {self.particles[VEL_X:VEL_X+1]}")
        tick = float(Config.get("bh", "tick_seconds"))
        
        #self.position += self.velocity * tick/2
        self.particles[:,POS_X:POS_X+1] += (self.particles[:,VEL_X:VEL_X+1] * (tick/2))
        self.particles[:,POS_Y:POS_Y+1] += (self.particles[:,VEL_Y:VEL_Y+1] * (tick/2))
        
        #self.velocity += self.acceleration * tick
        self.particles[:,VEL_X:VEL_X+1] += (self.particles[:,ACC_X:ACC_X+1] * tick)
        self.particles[:,VEL_Y:VEL_Y+1] += (self.particles[:,ACC_Y:ACC_Y+1] * tick)

        #self.position += self.velocity * tick/2
        self.particles[:,POS_X:POS_X+1] += (self.particles[:,VEL_X:VEL_X+1] * (tick/2))
        self.particles[:,POS_Y:POS_Y+1] += (self.particles[:,VEL_Y:VEL_Y+1] * (tick/2))

        #self.acceleration = np.zeros(2)
        self.particles[:,ACC_X:ACC_Y+1] = .0
