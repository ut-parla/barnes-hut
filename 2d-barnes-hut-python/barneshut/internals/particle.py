from .centreofmass import CentreOfMass
from .point import Point
from .vector import Vector
from . import constants as cn
import math

class Particle:
    particle_id_counter = 0

    def __init__(self):
        pass

    @staticmethod
    def from_line(line):
        fields = [float(x) for x in line.split(",")]
        p = Particle()
        p.pos      = Point(fields[0], fields[1])
        p.velocity = Point(fields[2], fields[3])
        p.mass = fields[4]
        p.accel = Vector()
        p.id = Particle.particle_id_counter
        Particle.particle_id_counter += 1
        return p

    def tick(self):
        # this looks wrong and I dont know why. Cant find a reliable source for this equation
        # this is from https://www.cs.utexas.edu/~rossbach/cs380p/lab/bh-submission-cs380p.html
        #self.pos.x += (cn.TICK_SECONDS * self.velocity.x) + (0.5 * self.accel.x * cn.TICK_SECONDS*cn.TICK_SECONDS)
        #self.pos.y += (cn.TICK_SECONDS * self.velocity.y) + (0.5 * self.accel.y * cn.TICK_SECONDS*cn.TICK_SECONDS)

        # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf

        self.pos.x += (self.velocity.x * cn.TICK_SECONDS/2) 
        self.pos.y += (self.velocity.y * cn.TICK_SECONDS/2)

        self.velocity.x += (self.accel.x * cn.TICK_SECONDS)
        self.velocity.y += (self.accel.y * cn.TICK_SECONDS)
        
        self.pos.x += (self.velocity.x * cn.TICK_SECONDS/2) 
        self.pos.y += (self.velocity.y * cn.TICK_SECONDS/2)
        
        self.accel = Vector()

    def getCentreOfMass(self):
        return CentreOfMass(self.mass, Point(self.pos.x, self.pos.y))

    def __repr__(self):
        return '<Particle x: {}, y:{}>'.format(self.pos.x, self.pos.y)
