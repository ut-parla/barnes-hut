import numpy as np

class Particle:

    def __init__(self, ar):
        self.particle = ar

    def get_array(self):
        return self.particle

    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")]
        part = np.ndarray(7)
        part[:5] = fields

        return Particle(part)
