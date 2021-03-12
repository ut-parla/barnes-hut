import numpy as np

particle_type = np.dtype([('px','f8'), ('py','f8'),   # particle position
                          ('mass','f8'),              # mass
                          ('vx','f8'), ('vy','f8'),   # velocity
                          ('ax','f8'), ('ay','f8'),   # acceleration
                          ('gx','f8'), ('gy','f8')])  # grid position
class Particle:

    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")] + [0]*4   #we need 4 fields (acc,gridpos), so *4
        return np.array(tuple(fields), dtype=particle_type)
