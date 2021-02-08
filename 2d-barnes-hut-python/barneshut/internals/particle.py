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
        self.velocity.x += self.accel.x
        self.velocity.y += self.accel.y
        self.accel = Vector()
        self.pos.x += cn.TICK_SECONDS * self.velocity.x / cn.SCALE_TO_METERS
        self.pos.y += cn.TICK_SECONDS * self.velocity.y / cn.SCALE_TO_METERS

    def getCentreOfMass(self):
        return CentreOfMass(self.mass, Point(self.pos.x, self.pos.y))

    def clamp(self, number, clampMax):
        val = number if number < clampMax else clampMax
        val = val if val > 0 else 0
        return math.floor(val)

    def __repr__(self):
        return '<Particle x: {}, y:{}>'.format(self.pos.x, self.pos.y)
