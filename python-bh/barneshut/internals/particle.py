import numpy as np

class Particle:

    def __init__(self, position, mass, velocity=(0,0)):
        self.position = np.array(position)
        self.mass = mass
        self.velocity = np.array(velocity)
        #self.acceleration = np.zeros(2)

    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")]
        return Particle(fields[0:2], fields[2], fields[3:5])
