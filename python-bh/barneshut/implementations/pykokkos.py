import logging
from math import sqrt, pow, ceil
from itertools import combinations, product
from typing import Tuple

import numpy as np
from numpy.lib import recfunctions as rfn
import pykokkos as pk

from .base import BaseBarnesHut
from barneshut.grid_decomposition import Box
from barneshut.internals.config import Config
from barneshut.internals.particle import Particle, particle_type
from timer import Timer

LEAF_OCCUPANCY = 0.7
N_DIM = 2

@pk.functor
class PyKokkosBox:
    def __init__(self, bottom_left: Tuple[int, int], top_right: Tuple[int, int]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.n = 0

        self.max_particles: int = int(Config.get("quadtree", "particles_per_leaf"))
        self.position: pk.View2D[pk.double] = pk.View([self.max_particles, N_DIM], pk.double)
        self.velocity: pk.View2D[pk.double] = pk.View([self.max_particles, N_DIM], pk.double)
        self.mass: pk.View1D[pk.double] = pk.View([self.max_particles], pk.double)
        self.acceleration: pk.View1D[pk.double] = pk.View([self.max_particles], pk.double)

        self.COM_initialized = False
        self.COM_position: pk.View2D[pk.double] = pk.View([2, N_DIM], pk.double)
        self.COM_velocity: pk.View2D[pk.double] = pk.View([2, N_DIM], pk.double)
        self.COM_mass: pk.View1D[pk.double] = pk.View([2], pk.double)
        self.COM_acceleration: pk.View1D[pk.double] = pk.View([2], pk.double)

    def is_empty(self) -> bool:
        return self.n == 0

    def is_full(self) -> bool:
        return self.n >= self.max_particles

    def add_particle(self, p: Particle) -> None:
        if self.is_full():
            print("adding to a full leaf, something is wrong")

        self.position[self.n] = [p["px"], p["py"]]
        self.velocity[self.n] = [p["vx"], p["vy"]]
        self.mass[self.n] = p["mass"]
        self.acceleration[self.n] = 0.0
        self.n += 1

    def get_COM(self):
        if not self.COM_initialized:
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            M = pk.parallel_reduce(self.n, self.COM_mass_kernel)
            x_avg = pk.parallel_reduce(self.n, self.COM_x_kernel) / M
            y_avg = pk.parallel_reduce(self.n, self.COM_y_kernel) / M
            self.init_COM((x_avg, y_avg), M)

    def init_COM(self, coords: Tuple[pk.double, pk.double], M: pk.double):
        self.COM_position[0][0] = coords[0]
        self.COM_position[0][1] = coords[1]
        self.COM_position[1][0] = 0
        self.COM_position[1][1] = 0
        self.COM_mass[0] = M

    @pk.workunit
    def COM_mass_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.mass[tid]
    
    @pk.workunit
    def COM_x_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.position[tid][0] * self.mass[tid]
        
    @pk.workunit
    def COM_y_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.position[tid][1] * self.mass[tid]

    def approximation_distance(self, other_box):
        corners1 = self.get_corners()
        corners2 = other_box.get_corners()

        # there's gotta be a better way to do this
        for x in corners1[0]:
            for y in corners1[1]:
                px1, px2 = corners2[0]
                py1, py2 = corners2[1]
                if (
                    ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
                    ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
                   ):
                    return False
        for x in corners2[0]:
            for y in corners2[1]:
                px1, px2 = corners1[0]
                py1, py2 = corners1[1]
                if (
                    ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
                    ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
                   ):
                    return False

        return True

class PyKokkosBarnesHut(BaseBarnesHut):
    """ PyKokkos implementation of nbody."""

    def __init__(self):
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
        self.grid = None

    def read_particles_from_file(self, filename):
        """Call the base class method and construct the kernels object"""

        super().read_particles_from_file(filename)
        # self.kernels = PyKokkosBarnesHutKernels(self.particles)

    def __next_perfect_square(self, n):
        if n%n**0.5 != 0:
            return pow( ceil(sqrt(n))  , 2)
        return n

    def __get_bounding_box(self):
        """ Get bounding box coordinates around all particles.
        Returns the bottom left and top right corner coordinates, making
        sure that it is a square.
        """
        # https://numpy.org/doc/stable/user/basics.rec.html#indexing-and-assignment-to-structured-arrays
        parts = rfn.structured_to_unstructured(self.particles[['px', 'py']], copy=False)
        max_x, max_y = np.max(parts, axis=0)[:2]
        min_x, min_y = np.min(parts, axis=0)[:2]

        x_edge, y_edge = max_x - min_x, max_y - min_y 
        if x_edge >= y_edge:
            max_y += (x_edge - y_edge)
        else:
            max_x += (y_edge - x_edge)

        assert (max_x-min_x)==(max_y-min_y)
        return (min_x, min_y), (max_x, max_y)

    def __create_grid(self, bottom_left, top_right, grid_dim):
        # x and y have the same edge length, so get x length
        step = (top_right[0]-bottom_left[0]) / grid_dim
        # create grid as a matrix, starting from bottom left
        self.grid = []
        logging.debug(f"Grid: {bottom_left}, {top_right}")
        for i in range(grid_dim):
            row = []
            for j in range(grid_dim):
                x = bottom_left[0] + (i*step)
                y = bottom_left[1] + (j*step)
                row.append(PyKokkosBox((x,y), (x+step, y+step)))
                logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        # get bounding box around all particles
        bb_min, bb_max = self.__get_bounding_box() 
        bottom_left = np.array(bb_min)
        top_right = np.array(bb_max)

        # if more than one particle per leaf, let's assume an occupancy of
        # 80% (arbitrary number), because if we use 100% we might have leaves
        # with >particles_per_leaf particles. This is all assuming a normal
        # random distribution.
        if self.particles_per_leaf == 1:
            nleaves = self.particles_per_leaf
        else:
            n = len(self.particles)
            nleaves = n / (LEAF_OCCUPANCY * self.particles_per_leaf)

        # find next perfect square
        nleaves = self.__next_perfect_square(nleaves)
        grid_dim = int(sqrt(nleaves))

        logging.debug(f'''With {LEAF_OCCUPANCY} occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {grid_dim}.
                Grid will be {grid_dim}x{grid_dim}''')
        
        self.__create_grid(bottom_left, top_right, grid_dim)

        # TODO: place this in a kernel
        # TODO: use numpy to do batches
        bb_x = np.array([bottom_left[0], top_right[0]])
        # bb_y = np.array([bottom_left[1], top_right[1]])
        edge_len = bb_x[1] - bb_x[0]
        step =  edge_len / grid_dim 

        # placements is an array mapping points to their position in the matrix
        # this is just so we can easily map to numpy/cuda later
        placements = np.ndarray((len(self.particles), 2))
        for i, p in enumerate(self.particles):
            x, y = p['px'], p['py']
            px, py = x/step, y/step
            placements[i][0], placements[i][1] = px, py 

        # if we need to sort:
        # a = np.array([(1,4,5), (2,1,1), (3,5,1)], dtype='f8, f8, f8')
        #   np.argsort(a, order=('f1', 'f2'))
        # or a.sort(...)

        for i, p in enumerate(placements):
            # need to get min because of float rounding
            x = min(int(p[0]), grid_dim-1)
            y = min(int(p[1]), grid_dim-1)
            # logging.debug(f"adding point {i} ({self.particles[i].position}) to box {x}/{y}")
            self.grid[x][y].add_particle(self.particles[i])

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):
        n = len(self.grid)
        # do all distinct pairs interaction
        cells = product(range(n), range(n))
        pairs = combinations(cells, 2)

        for p1, p2 in pairs:
            l1 = self.grid[p1[0]][p1[1]]
            l2 = self.grid[p2[0]][p2[1]]
            l1.apply_force(l2)

        # and all self to self interaction
        for l in range(n):
            leaf = self.grid[l][l]
            leaf.apply_force(leaf)

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].tick()

    def run(self, n_iterations, partitions=None, print_particles=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""
        with Timer.get_handle("end-to-end"):
            for _ in range(n_iterations):
                # Step 1: create tree (if Barnes-Hut), group-by points by box (if Decomposition)
                with Timer.get_handle("tree-creation"):
                    self.create_tree()

                # Step 2: summarize.
                with Timer.get_handle("summarization"):
                    self.summarize()

                # # Step 3: evaluate.
                # with Timer.get_handle("evaluation"):
                #     self.evaluate()

                # # Step 4: tick particles using timestep
                # with Timer.get_handle("timestep"):
                #     self.timestep()

        Timer.print()

    def print_particles(self):
        """Print all particles' coordinates for debugging"""
        #for p in self.particles:
        #    print(repr(p))
        # TODO
        raise NotImplementedError()