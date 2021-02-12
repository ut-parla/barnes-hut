from __future__ import annotations
from numba import f4, i4
from numba.experimental import jitclass
import numpy as np


#numba spec
spec = [
    ('pX', f4), ('pY', f4),
    ('mass', i4)
]

#@jitclass(spec)
class CentreOfMass:

    def __init__(self, pX, pY, mass):
        self.pX = pX
        self.pY = pY
        self.mass = mass

    def combine(self, other: CentreOfMass):
        if (other is None):
            return

        m = self.mass + other.mass
        if m == 0:
            print("ZERO")

        newX = np.divide(self.pX * self.mass + other.pX * other.mass, m)
        newY = np.divide(self.pY * self.mass + other.pY * other.mass, m)

        self.pX = newX
        self.pY = newY
        self.mass = m

    def __repr__(self):
        return '<CentreOfMass mass: {}, pos:{},{}>'.format(self.mass, self.pX, self.pY)
