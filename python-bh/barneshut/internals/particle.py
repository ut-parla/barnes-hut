import numpy as np

particle_type = np.dtype([('px','f8'), ('py','f8'), ('mass','f8'), ('vx','f8'), ('vy','f8')]) 
class Particle:

    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")]
        return np.array(tuple(fields), dtype=particle_type)
