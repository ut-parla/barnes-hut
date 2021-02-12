from . import constants as cn
import numpy as np
from numba import f4, i4
from numba.experimental import jitclass

#numba spec
spec = [
#TBD
]

def particle_from_line(line):
    fields = [float(x) for x in line.split(",")]
    return Particle(*fields)

#@jitclass(spec)
class Particle:

    def __init__(self, position, mass, velocity=(0,0)):
        self.position = np.array(position)
        self.mass = mass
        self.velocity = np.array(velocity)
        self.acceleration = np.zeros(2)

    def calculate_distance(self, other):
        return np.linalg.norm(self.position - other.position)

    # G = 6.673 x 10-11 Nm^2/kg^2
    # Fgrav = (G*m1*m2)/d^2
    # F = m*a
    def apply_force(self, other, isCOM=False):
        diff = self.position - other.position
        dist = np.linalg.norm(diff)
        f = (cn.GRAVITATIONAL_CONSTANT * self.mass * other.mass) / (dist*dist)

        # update self acceleration
        self.acceleration -= (f * diff) / self.mass
        # update other particles acceleration
        if not isCOM:
            other.acceleration += (f * diff) / self.mass

    def tick(self):
        # this looks wrong and I dont know why. Cant find a reliable source for this equation
        # this is from https://www.cs.utexas.edu/~rossbach/cs380p/lab/bh-submission-cs380p.html
        #self.pos.x += (cn.TICK_SECONDS * self.velocity.x) + (0.5 * self.accel.x * cn.TICK_SECONDS*cn.TICK_SECONDS)
        #self.pos.y += (cn.TICK_SECONDS * self.velocity.y) + (0.5 * self.accel.y * cn.TICK_SECONDS*cn.TICK_SECONDS)

        # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf
        self.position += self.velocity * cn.TICK_SECONDS/2
        self.velocity += self.acceleration * cn.TICK_SECONDS
        self.position += self.velocity * cn.TICK_SECONDS/2
        self.acceleration = np.zeros(2)
    
    def __repr__(self):
        return '<Particle x: {}, y:{}>'.format(*self.position)
