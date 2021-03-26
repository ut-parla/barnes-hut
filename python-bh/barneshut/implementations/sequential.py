import logging
import numpy as np
from barneshut.grid_decomposition import Box
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy import sqrt
from timer import Timer
from .base import BaseBarnesHut
import barneshut.internals.particle as p

class SequentialBarnesHut (BaseBarnesHut):
    """ Sequential implementation of nbody. Currently not Barnes-hut but
    a box decomposition."""

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None

        # setup functions
        from barneshut.kernels.grid_decomposition.sequential.grid import \
            get_grid_placement_fn
        self.__grid_placement = get_grid_placement_fn()

        from barneshut.kernels.grid_decomposition.sequential.evaluation import \
            get_evaluation_fn
        self.__evaluate = get_evaluation_fn()
        
        from barneshut.kernels.gravity import get_gravity_kernel
        self.__grav_kernel = get_gravity_kernel()

    def create_grid_boxes(self):
        """Use bounding boxes coordinates, create the grid
        matrix and their boxes
        """
        # x and y have the same edge length, so get x length
        step = (self.max_xy[0]-self.min_xy[0]) / self.grid_dim
        # create grid as a matrix, starting from bottom left
        self.grid = []
        logging.debug(f"Grid: {self.min_xy}, {self.max_xy}")
        for i in range(self.grid_dim):
            row = []
            for j in range(self.grid_dim):
                x = self.min_xy[0] + (i*step)
                y = self.min_xy[1] + (j*step)
                row.append(Box((x,y), (x+step, y+step), grav_kernel=self.__grav_kernel))
                #logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()
        self.create_grid_boxes()

        # call kernel to place points
        # grid placements sets gx, gy to their correct place in the grid
        self.particles = self.__grid_placement(self.particles, self.min_xy, self.step, self.grid_dim)

        # sort particles by grid position
        self.particles.view(p.fieldsstr).sort(order=[p.gxf, p.gyf], axis=0)

        coords, lens = np.unique(self.particles[:, p.gx:p.gy+1], return_index=True, axis=0)
        coords = coords.astype(int)
        ncoords = len(coords)
        added = 0

        for i in range(ncoords):
            x,y = np.clip(coords[i], 0, self.grid_dim-1)
            start = lens[i]
            # if last, get remaining
            end = lens[i+1] if i < ncoords-1 else len(self.particles)

            added += end-start
            #logging.debug(f"adding {end-start} particles to box {x}/{y}")

            self.grid[x][y].add_particle_slice(self.particles[start:end])

        #logging.debug(f"added {added} total particles")
        assert added == len(self.particles)

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):
        self.__evaluate(self.grid)

        print("after ", self.particles[:, p.ax:p.ay+1])

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                if not self.grid[i][j].cloud.is_empty():
                    self.grid[i][j].tick()

    def ensure_particles_id_ordered(self):
        # just sort by id since they were shuffled
        self.particles.view(p.fieldsstr).sort(order=p.idf, axis=0)

        print("sort ", self.particles[:, p.ax:p.ay+1])

    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i]
            return samples