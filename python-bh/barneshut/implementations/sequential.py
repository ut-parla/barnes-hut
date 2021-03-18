import logging
import numpy as np
from barneshut.grid_decomposition import Box
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy import sqrt
from numpy.lib.recfunctions import structured_to_unstructured as unst
from timer import Timer
from .base import BaseBarnesHut

LEAF_OCCUPANCY = 0.7

class SequentialBarnesHut (BaseBarnesHut):
    """ Sequential implementation of nbody. Currently not Barnes-hut but
    a box decomposition."""

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
        self.grid = None
        self.particles_argsort = None

        # setup functions
        from barneshut.kernels.grid_decomposition.sequential.grid import \
            get_grid_placement_fn
        self.__grid_placement = get_grid_placement_fn()

        from barneshut.kernels.grid_decomposition.sequential.evaluation import \
            get_evaluation_fn
        self.__evaluate = get_evaluation_fn()
        
    def __alloc_grid(self, bottom_left, top_right, grid_dim):
        """Use bounding boxes coordinates, create the grid
        matrix and their boxes
        """
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
                row.append(Box((x,y), (x+step, y+step)))
                logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)

    def __create_grid(self):
        # get square bounding box around all particles
        unstr_points = unst(self.particles[['px', 'py']], copy=False)
        bb_min, bb_max = get_bounding_box(unstr_points)
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
        nleaves = next_perfect_square(nleaves)
        self.grid_dim = int(sqrt(nleaves))
        logging.debug(f'''With {LEAF_OCCUPANCY} occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {self.grid_dim}.
                Grid will be {self.grid_dim}x{self.grid_dim}''')
        
        # create grid matrix
        self.__alloc_grid(bottom_left, top_right, self.grid_dim)
        self.step = (top_right[0] - bottom_left[0]) / self.grid_dim 
        self.min_xy = bottom_left
        
    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i].copy()
            return samples

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.__create_grid()        

        # call kernel to place points
        with Timer.get_handle("placement_kernel"):
            # grid placements sets gx, gy to their correct place in the grid
            self.particles = self.__grid_placement(self.particles, self.min_xy, self.step, self.grid_dim)

            # if we are checking accuracy, we need to save how we sorted particles.
            # performance doesn't matter, so do the easy way
            if self.checking_accuracy:
                self.particles_argsort = np.argsort(self.particles, order=('gx', 'gy'), axis=0)

            # sort by grid position
            self.particles.sort(order=('gx', 'gy'), axis=0)
            up = unst(self.particles)
            # TODO: change from unique to a manual O(n) scan, we can do it
            coords, lens = np.unique(up[:, 7:9], return_index=True, axis=0)
            coords = coords.astype(int)
            ncoords = len(coords)
            added = 0

            for i in range(ncoords):
                x,y = coords[i]
                start = lens[i]
                # if last, get remaining
                end = lens[i+1] if i < ncoords-1 else len(self.particles)

                added += end-start
                logging.debug(f"adding {end-start} particles to box {x}/{y}")
                self.grid[x][y].add_particle_slice(self.particles[start:end])

            logging.debug(f"added {added} total particles")
            assert added == len(self.particles)

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):
        self.__evaluate(self.grid)

        # if checking accuracy, unsort the particles
        if self.checking_accuracy:
            self.particles = self.particles[self.particles_argsort]
            self.particles_argsort = None

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                if not self.grid[i][j].cloud.is_empty():
                    self.grid[i][j].tick()
