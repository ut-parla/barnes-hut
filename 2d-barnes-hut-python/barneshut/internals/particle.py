from .centreofmass import CentreOfMass
from . import constants as cn
import math
import numpy as np
from numba import f4, i4
from numba.experimental import jitclass

#numba spec
spec = [
    ('pX', f4), ('pY', f4),
    ('vX', f4), ('vY', f4),
    ('aX', f4), ('aY', f4),
    ('mass', i4)
]

def particle_from_line(line):
    fields = [float(x) for x in line.split(",")]
    return Particle(*fields)

#@jitclass(spec)
class Particle:

    def __init__(self, pX, pY, vX, vY, mass):
        self.pX = pX
        self.pY = pY
        self.vX = vX
        self.vY = vY
        self.mass = mass
        self.aX = 0
        self.aY = 0

    def calculate_distance(self, oX, oY):
        x = math.fabs(self.pX - oX)
        y = math.fabs(self.pY - oY)
        return math.hypot(x, y)

    def apply_force(self, p2):
        # G = 6.673 x 10-11 Nm^2/kg^2
        # Fgrav = (G*m1*m2)/d^2
        # F = m*a
        xDiff = self.pX - p2.pX 
        yDiff = self.pY - p2.pY 
        dist = self.calculate_distance(p2.pX, p2.pY)

        #f = constants.TICK_SECONDS * (constants.GRAVITATIONAL_CONSTANT * self.mass * p2.mass) / dist*dist
        f = np.divide(cn.GRAVITATIONAL_CONSTANT * self.mass * p2.mass, dist*dist*dist)

        self.aX -= np.divide(f * xDiff, self.mass)
        self.aY -= np.divide(f * yDiff, self.mass)

        p2.aX += np.divide(f * xDiff, p2.mass)
        p2.aY += np.divide(f * yDiff, p2.mass)

    def apply_force_COM(self, com):
        # G = 6.673 x 10-11 Nm^2/kg^2
        # Fgrav = (G*m1*m2)/d^2
        # F = m*a
        xDiff = self.pX - com.pX
        yDiff = self.pY - com.pY 
        dist = self.calculate_distance(com.pX, com.pY)

        #f = constants.TICK_SECONDS * (constants.GRAVITATIONAL_CONSTANT * self.mass * com.mass) / dist*dist
        f = np.divide(cn.GRAVITATIONAL_CONSTANT * self.mass * com.mass, dist*dist*dist)

        self.aX -= np.divide(f * xDiff, self.mass)
        self.aY -=np.divide( f * yDiff, self.mass)


    def tick(self):
        # this looks wrong and I dont know why. Cant find a reliable source for this equation
        # this is from https://www.cs.utexas.edu/~rossbach/cs380p/lab/bh-submission-cs380p.html
        #self.pos.x += (cn.TICK_SECONDS * self.velocity.x) + (0.5 * self.accel.x * cn.TICK_SECONDS*cn.TICK_SECONDS)
        #self.pos.y += (cn.TICK_SECONDS * self.velocity.y) + (0.5 * self.accel.y * cn.TICK_SECONDS*cn.TICK_SECONDS)

        # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf

        self.pX += (self.vX * cn.TICK_SECONDS/2) 
        self.pY += (self.vY * cn.TICK_SECONDS/2)

        self.vX += (self.aX * cn.TICK_SECONDS)
        self.vY += (self.aY * cn.TICK_SECONDS)
        
        self.pX += (self.vX * cn.TICK_SECONDS/2) 
        self.pY += (self.vY * cn.TICK_SECONDS/2)
        
        self.aX = 0
        self.aY = 0

    def getCentreOfMass(self):
        return CentreOfMass(self.pX, self.pY, self.mass)

    def __repr__(self):
        return '<Particle x: {}, y:{}>'.format(self.pX, self.pY)
