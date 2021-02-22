from . import Particle
from .indices import *
from .config import Config
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg.blas import zhpr, dspr2, zhpmv
from numpy.linalg import norm

class Cloud:

    def __init__(self, initd_array=None):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.COM = None

        # default constructor, in which the cloud has no particles
        if initd_array is None:
            self.particle_array = np.ndarray((self.max_particles, 7))
            self.n = 0
        # create the cloud with some particles, like a concatenation of two clouds
        else:
            self.particle_array = initd_array
            self.n = initd_array.shape[0]

        # TODO: this is not the best place to do this
        fc = Config.get("general", "force_calculation")
        if fc == "p2cloud":
            self.__apply_force_c2c = self.__apply_force_c2c_particle2cloudloop
        elif fc == "vect":
            self.__apply_force_c2c = self.__apply_force_c2c_vect
        elif fc == "blas":
            self.__apply_force_c2c = self.__apply_force_c2c_blas

    @property
    def particles(self):
        return self.particle_array[:self.n]

    @property
    def positions(self):
        return self.particle_array[:self.n,POS_X:POS_Y+1]

    @property
    def masses(self):
        return self.particle_array[:self.n,MASS:MASS+1]

    @property
    def accelerations(self):
        return self.particle_array[:self.n,ACC_X:ACC_Y+1]

    @accelerations.setter
    def accelerations(self, acc):
        self.particle_array[:self.n,ACC_X:ACC_Y+1] = acc

    @property
    def velocities(self):
        return self.particle_array[:self.n,VEL_X:VEL_Y+1]

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.max_particles

    def add_particle(self, part):
        self.particle_array[self.n] = part
        self.n += 1

    def concatenation(self, other_cloud):
        return Cloud( np.concatenate((self.particles, other_cloud.particles)) )

    def get_COM(self):
        if self.COM is None:
            #ppm = np.multiply.reduce(self.particles[:MASS], axis=1)
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            mx = np.multiply(self.particles[:,POS_X:POS_X+1], self.particles[:,MASS:MASS+1])
            my = np.multiply(self.particles[:,POS_Y:POS_Y+1], self.particles[:,MASS:MASS+1])
            M  = np.sum(self.particles[:, MASS:MASS+1])
            #print(f"COM: {mx}\n{my}\n{M}")

            newX = np.divide(np.sum(mx), M)
            newY = np.divide(np.sum(my), M)

            self.COM = np.zeros(7)
            self.COM[:3] = [newX, newY, M]

        return self.COM

    def apply_force(self, other_set, is_COM=False):
        G = float(Config.get("bh", "grav_constant"))
        if not is_COM:
            self.__apply_force_c2c(other_set)
        else:
            com = other_set.get_COM()
            for p1 in self.particles:
                diff = p1[:POS_Y+1] - com[:POS_Y+1]
                dist = np.linalg.norm(diff)
                f = np.divide(G * p1[MASS:MASS+1] * com[MASS:MASS+1], dist*dist)
                p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]

    # this is a somewhat naive approach using numpy
    def __apply_force_c2c_particle2cloudloop(self, other_cloud):
        G = float(Config.get("bh", "grav_constant"))
        for p in self.particles:
            p2 = other_cloud.particles
            
            #diff = p1[:POS_Y+1] - p2[:POS_Y+1]
            p2[:, POS_X:POS_Y+1] -= p[POS_X:POS_Y+1]
            
            pp2 = np.concatenate(([p], p2))
            dist_matrix = squareform(pdist(pp2)) 

            # f = (G * p1[MASS:MASS+1] * p2[MASS:MASS+1]) / (dist*dist)
            f = G * p[MASS:MASS+1] * p2[:, MASS:MASS+1]
            # slice: ignore first row since it's p itself, and only use first column, which is
            # pairwise distance of p to all points in p2
            f /= dist_matrix[1:,:1]

            #p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]
            # calculate forces from all of p2 to p
            p[ACC_X:ACC_Y+1]    -= np.add.reduce((f*p2[:, POS_X:POS_Y+1]) / p[MASS:MASS+1])
            # calculate forces from p to all of p2
            p2[:,ACC_X:ACC_Y+1] += (f * p2[:, POS_X:POS_Y+1]) / p2[:,MASS:MASS+1]
       
       
    # calculations below (pmende, vect and blas) are from this source:
    # https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python
    def __apply_force_c2c_vect(self, other_cloud):
        # get G, positions and masses of concatenation
        G = float(Config.get("bh", "grav_constant"))
        cc = self.concatenation(other_cloud)        
        masses = cc.masses
        positions = cc.positions

        # actual calculation
        mass_matrix = masses.reshape((1, -1, 1))*masses.reshape((-1, 1, 1))
        disps = positions.reshape((1, -1, 2)) - positions.reshape((-1, 1, 2)) # displacements
        dists = norm(disps, axis=2)
        dists[dists == 0] = 1 # Avoid divide by zero warnings
        forces = G*disps*mass_matrix/np.expand_dims(dists, 2)**3
        acc = forces.sum(axis=1)/masses.reshape(-1, 1)

        # add accelerations
        self.accelerations        += acc[:self.n,:]
        other_cloud.accelerations += acc[self.n:,:]


    def __apply_force_c2c_blas(self, other_cloud):
        # get G, positions and masses of concatenation
        G = float(Config.get("bh", "grav_constant"))
        cc = self.concatenation(other_cloud)        
        mas = cc.masses
        pos = cc.positions

        n = mas.size
        # trick: use complex Hermitian to get the packed anti-symmetric
        # outer difference in the imaginary part of the zhpr answer
        # don't want to sum over dimensions yet, therefore must do them one-by-one
        trck = np.zeros((3, n * (n + 1) // 2), complex)
        for a, p in zip(trck, pos.T - 1j):
            zhpr(n, -2, p, a, 1, 0, 0, 1)
            # does  a  ->  a + alpha x x^H
            # parameters: n             --  matrix dimension
            #             alpha         --  real scalar
            #             x             --  complex vector
            #             ap            --  packed Hermitian n x n matrix a
            #                               i.e. an n(n+1)/2 vector
            #             incx          --  x stride
            #             offx          --  x offset
            #             lower         --  is storage of ap lower or upper
            #             overwrite_ap  --  whether to change a inplace
        # as a by-product we get pos pos^T:
        ppT = trck.real.sum(0) + 6
        # now compute matrix of squared distances ...
        # ... using (A-B)^2 = A^2 + B^2 - 2AB
        # ... that and the outer sum X (+) X.T equals X ones^T + ones X^T
        dspr2(n, -0.5, ppT[np.r_[0, 2:n+1].cumsum()], np.ones((n,)), ppT,
            1, 0, 1, 0, 0, 1)
        # does  a  ->  a + alpha x y^T + alpha y x^T    in packed symmetric storage
        # scale anti-symmetric differences by distance^-3
        np.divide(trck.imag, ppT*np.sqrt(ppT), where=ppT.astype(bool),
                out=trck.imag)
        # it remains to scale by mass and sum
        # this can be done by matrix multiplication with the vector of masses ...
        # ... unfortunately because we need anti-symmetry we need to work
        # with Hermitian storage, i.e. complex numbers, even though the actual
        # computation is only real:
        out = np.zeros((2, n), complex)
        for a, o in zip(trck, out):
            zhpmv(n, 0.5, a, mas*-1j, 1, 0, 0, o, 1, 0, 0, 1)
            # multiplies packed Hermitian matrix by vector
        acc = out.real.T
    
        # add accelerations
        self.accelerations        += acc[:self.n,:]
        other_cloud.accelerations += acc[self.n:,:]

            
    def tick_particles(self):
        # current equations are from 3 step integrator from https://www.maths.tcd.ie/~btyrrel/nbody.pdf
        #print(f"TICK: POS: {self.particles[POS_X:POS_X+1]}\nVEL: {self.particles[VEL_X:VEL_X+1]}")
        tick = float(Config.get("bh", "tick_seconds"))
        
        #self.position += self.velocity * tick/2
        self.particles[:,POS_X:POS_X+1] += (self.particles[:,VEL_X:VEL_X+1] * (tick/2))
        self.particles[:,POS_Y:POS_Y+1] += (self.particles[:,VEL_Y:VEL_Y+1] * (tick/2))
        
        #self.velocity += self.acceleration * tick
        self.particles[:,VEL_X:VEL_X+1] += (self.particles[:,ACC_X:ACC_X+1] * tick)
        self.particles[:,VEL_Y:VEL_Y+1] += (self.particles[:,ACC_Y:ACC_Y+1] * tick)

        #self.position += self.velocity * tick/2
        self.particles[:,POS_X:POS_X+1] += (self.particles[:,VEL_X:VEL_X+1] * (tick/2))
        self.particles[:,POS_Y:POS_Y+1] += (self.particles[:,VEL_Y:VEL_Y+1] * (tick/2))

        #self.acceleration = np.zeros(2)
        self.particles[:,ACC_X:ACC_Y+1] = .0
