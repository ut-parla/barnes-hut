import logging
import numpy as np
from barneshut.grid_decomposition import Box
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy import sqrt
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

        from barneshut.kernels.grid_decomposition.sequential.evaluation import \
            get_evaluation_fn
        self.__evaluate = get_evaluation_fn()
        
    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()
        self.create_grid_boxes()

        self.particles[:, p.gx:p.gy+1] = self.particles[:, p.px:p.py+1]
        self.particles[:, p.gx:p.gy+1] = (self.particles[:, p.gx:p.gy+1] - self.min_xy) / self.step
        self.particles[:, p.gx:p.gy+1] = np.clip(np.floor(self.particles[:, p.gx:p.gy+1]), 0, self.grid_dim-1)

        # sort particles by grid position
        self.particles.view(p.fieldsstr).sort(order=[p.gxf, p.gyf], axis=0, kind="stable")

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

        assert added == len(self.particles)

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):   
        self.__evaluate(self.grid)

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                if not self.grid[i][j].cloud.is_empty():
                    self.grid[i][j].tick()

    def ensure_particles_id_ordered(self):
        # just sort by id since they were shuffled
        self.particles.view(p.fieldsstr).sort(order=p.idf, axis=0)

    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i]
            return samples