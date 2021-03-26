import numpy as np

px, py = 0, 1
mass = 2
vx, vy = 3, 4
ax, ay = 5, 6
gx, gy = 7, 8
id = 9

gxf, gyf = 'f7', 'f8'
massf = 'f2'
idf = 'f9'
nfields = 10
fieldsstr = 'f8,f8,f8,f8,f8,f8,f8,f8,f8,f8'
ftype = np.float64

# structured arrays was likely a bad choice.. it seems random
# when we get a slice or a copy...
# particle_type = np.dtype([('px','f8'), ('py','f8'),   # particle position
#                           ('mass','f8'),              # mass
#                           ('vx','f8'), ('vy','f8'),   # velocity
#                           ('ax','f8'), ('ay','f8'),   # acceleration
#                           ('gx','f8'), ('gy','f8')])  # grid position
class Particle:
    id_counter = 0.0
    @staticmethod
    def particle_from_line(line):
        fields = [float(x) for x in line.split(",")] + [.0]*4 + [Particle.id_counter]   #we need 4 fields (acc,gridpos), so *4
        Particle.id_counter += 1
        return fields