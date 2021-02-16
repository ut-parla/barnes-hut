import numpy as np

class Particle(np.ndarray):

    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")]
        part = Particle(7)
        part[:5] = fields
        return part
