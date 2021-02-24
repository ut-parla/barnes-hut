from . import Particle
from .indices import *
from .config import Config
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg.blas import zhpr, dspr2, zhpmv
from numpy.linalg import norm
from .guvectorize_ops import *
from functools import partialmethod

# hopefully it won't be too hard to switch from 2d to 3d
N_DIM = 2

class Cloud:

    def __init__(self, com_particle=None, concatenation=None):
        self.max_particles = int(Config.get("quadtree", "particles_per_leaf"))
        self.COM = None

        # If we wanna get a cloud representation of a COM
        if com_particle is not None:
            self.n = 1
            # we allocate an extra element because of guvectorize cuda stuff
            self.__positions = np.array([com_particle.position, (0,0)])
            self.__masses = np.array([com_particle.mass,0.0])
            self.__velocities = np.ndarray((2, N_DIM))
            self.__accelerations = np.ndarray((2, N_DIM))
        # want to concatenate clouds
        elif concatenation is not None:
            c1, c2 = concatenation
            self.n = c1.n + c2.n
            self.__positions = np.concatenate((c1.positions, c2.positions))
            self.__masses = np.concatenate((c1.masses, c2.masses))
            self.__velocities = np.ndarray((self.n, N_DIM))
            self.__accelerations = np.ndarray((self.n, N_DIM))
        # if this is an empty Cloud creation
        else:
            self.n = 0
            # add one to max because of guvectorize cuda stuff
            self.__positions = np.ndarray((self.max_particles+1, N_DIM))
            self.__masses = np.ndarray(self.max_particles+1)
            self.__velocities = np.ndarray((self.max_particles+1, N_DIM))
            self.__accelerations = np.ndarray((self.max_particles+1, N_DIM))

        # TODO: this is not the best place/way to do this
        fc = Config.get("general", "force_calculation")
        if fc == "p2cloud":
            self.__apply_force = self.__apply_force_particle2cloudloop
        elif fc == "vect":
            self.__apply_force = self.__apply_force_vect
        elif fc == "blas":
            #print("results using blas are not correct. this is simply an upper bound on performance")
            self.__apply_force = self.__apply_force_blas
        # TODO: I tried using partialmethod here but it didnt work
        elif fc == "guvectorize-cpu":
            self.__apply_force = self.__apply_force_guvectorize_cpu
        elif fc == "guvectorize-parallel":
            self.__apply_force = self.__apply_force_guvectorize_parallel
        elif fc == "guvectorize-cuda":
            self.__apply_force = self.__apply_force_guvectorize_cuda

    #
    # general getter/setters
    #
    @property
    def positions(self):
        return self.__positions[:self.n]

    @positions.setter
    def positions(self, pos):
        self.__positions[:self.n] = pos

    @property
    def velocities(self):
        return self.__velocities[:self.n]

    @positions.setter
    def velocities(self, v):
        self.__velocities[:self.n] = v

    @property
    def masses(self):
        return self.__masses[:self.n]
    
    @property
    def accelerations(self):
        return self.__accelerations[:self.n]

    @accelerations.setter
    def accelerations(self, acc):
        self.__accelerations[:self.n] = acc

    def is_empty(self):
        return self.n == 0

    def is_full(self):
        return self.n >= self.max_particles

    def add_particle(self, part: Particle):
        self.__positions[self.n] = part.position
        self.__velocities[self.n] = part.velocity
        self.__masses[self.n] = part.mass
        self.__accelerations[self.n] = 0.0
        self.n += 1

    def concatenation(self, other_cloud):
        return Cloud(concatenation=(self, other_cloud))

    def get_COM(self):
        if self.COM is None:
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            M = np.sum(self.masses)
            coords = np.multiply(self.positions, self.masses[:, np.newaxis])
            coords = np.add.reduce(coords)
            coords /= M
            self.COM = Cloud(com_particle=Particle(coords, M))
        return self.COM

    def tick_particles(self):
        tick = float(Config.get("bh", "tick_seconds"))
        self.velocities += self.accelerations * tick
        self.accelerations[:,:] = 0.0               
        self.positions += self.velocities * tick

    def apply_force(self, other_cloud, use_COM=False):
        #if use_COM then we don't do all p2p computation, instead we get the COM of the cloud
        other = other_cloud if use_COM is False else other_cloud.get_COM()
        # pass use_COM in case the calculation needs to know if other is a COM or a set of particles
        self.__apply_force(other, use_COM)

    def __apply_force_guvectorize_cuda(self, other_cloud, __is_COM):
        G = float(Config.get("bh", "grav_constant"))
        # let's do the biggest set as first parameter, since guvectorize parallelize based on it's shape
        if self.n >= other_cloud.n:
            c1, c2 = self, other_cloud
        else:
            c1, c2 = other_cloud, self
        #print(f"c1 has {c1.n} particles, c2 has {c2.n}")

        # hack to change the dimension of output array, since v2 puts the c1 particle accelerations
        # at the last element of each output matrix.
        # this is why there's this whole slicing magic afterwards
        c2.n += 1
        accs = guvect_point_to_cloud_cuda(c1.positions, c1.masses, c2.positions, c2.masses, G)
        c2.n -= 1

        c2_sliced = accs[:,:-1]
        c2_acc = np.add.reduce(c2_sliced)        
        #print(f"c2_acc\n{c2_acc}")

        n = c2.n
        #print(f"accs\n{accs}")
        c1_sliced = accs[:, n::n]
        #print(f"c1_sliced:\n{c1_sliced}\n")
        c1_acc = c1_sliced.squeeze(axis=1)
        #print(f"c1_acc:\n{c1_acc}\n")
    
        c1.accelerations += c1_acc
        c2.accelerations += np.add.reduce(c2_acc)   
       
    def __apply_force_guvectorize_host(self, other_cloud, __is_COM, func):
        G = float(Config.get("bh", "grav_constant"))
        # this is really cool. the signature of this function takes a single point, but we pass multiple points instead
        # this way numpy can vectorize these operations.
        # let's do the biggest set as first parameter, since guvectorize parallelize based on it's shape
        if self.n >= other_cloud.n:
            c1, c2 = self, other_cloud
        else:
            c1, c2 = other_cloud, self
        c1_acc, c2_acc = func(c1.positions, c1.masses, c2.positions, c2.masses, G)
        c1.accelerations += c1_acc
        c2.accelerations += np.add.reduce(c2_acc)

    def __apply_force_guvectorize_cpu(self, other_cloud, __is_COM):
        self.__apply_force_guvectorize_host(other_cloud, __is_COM, guvect_point_to_cloud_cpu)

    def __apply_force_guvectorize_parallel(self, other_cloud, __is_COM):
        self.__apply_force_guvectorize_host(other_cloud, __is_COM, guvect_point_to_cloud_parallel)

    # calculations below (pmende, vect and blas) are from this source:
    # https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python
    def __apply_force_vect(self, other_cloud, __is_COM):
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

    def __apply_force_blas(self, other_cloud, __is_COM):
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

    # this is a somewhat naive approach using numpy
    def __apply_force_particle2cloudloop(self, other_cloud):
        # this has totally unnaceptable performance so I didnt fix it
        raise("NYI")

        # G = float(Config.get("bh", "grav_constant"))
        # for p in self.particles:
        #     p2 = other_cloud.particles
            
        #     #diff = p1[:POS_Y+1] - p2[:POS_Y+1]
        #     p2[:, POS_X:POS_Y+1] -= p[POS_X:POS_Y+1]
            
        #     pp2 = np.concatenate(([p], p2))
        #     dist_matrix = squareform(pdist(pp2)) 

        #     # f = (G * p1[MASS:MASS+1] * p2[MASS:MASS+1]) / (dist*dist)
        #     f = G * p[MASS:MASS+1] * p2[:, MASS:MASS+1]
        #     # slice: ignore first row since it's p itself, and only use first column, which is
        #     # pairwise distance of p to all points in p2
        #     f /= dist_matrix[1:,:1]

        #     #p1[ACC_X:ACC_Y+1] -= (f * diff) / p1[MASS:MASS+1]
        #     # calculate forces from all of p2 to p
        #     p[ACC_X:ACC_Y+1]    -= np.add.reduce((f*p2[:, POS_X:POS_Y+1]) / p[MASS:MASS+1])
        #     # calculate forces from p to all of p2
        #     p2[:,ACC_X:ACC_Y+1] += (f * p2[:, POS_X:POS_Y+1]) / p2[:,MASS:MASS+1]
