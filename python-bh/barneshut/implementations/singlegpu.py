import logging
import numpy as np
from barneshut.internals.config import Config
import barneshut.internals.particle as p
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from math import sqrt, ceil
from .base import BaseBarnesHut
from numba import cuda

#import kernels
from barneshut.kernels.grid_decomposition.gpu.grid import *

LEAF_OCCUPANCY = 0.7

# TODO: improve this thing to be an optimal
THREADS_PER_BLOCK = 128

class SingleGPUBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None
        self.G = float(Config.get("bh", "grav_constant"))
        self.device_arrays_initd = False
        self.debug = logging.root.level == logging.DEBUG
        
    def __init_device_arrays(self):
        """ Allocate arrays on the device and copy the particles
        array too. This is done only once.
        """
        if not self.device_arrays_initd:
            # alloc gpu arrays
            #self.d_grid_box_count = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_grid_box_cumm  = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int64)
            self.d_particles      = cuda.to_device(self.particles)
            self.d_COMs           = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float)
            self.device_arrays_initd = True

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()        
        self.__init_device_arrays()

        self.d_particles_sort = cuda.device_array_like(self.particles)
        # copy a zero'd out matrix
        zz = np.zeros((self.grid_dim,self.grid_dim), dtype=np.int32)
        self.d_grid_box_count = cuda.to_device(zz)

        #call kernel
        blocks = ceil(self.n_particles / THREADS_PER_BLOCK)
        threads = THREADS_PER_BLOCK
        g_place_particles[blocks, threads](self.d_particles, self.min_xy, self.step,
                          self.grid_dim, self.d_grid_box_count)
        g_calculate_box_cumm[1, 1](self.grid_dim, self.d_grid_box_count, self.d_grid_box_cumm)        

        g_sort_particles[blocks, threads](self.d_particles, self.d_particles_sort, self.d_grid_box_count)

        # Swap so original d_particles is deallocated. dealloc d_grid_box_count
        self.d_particles = self.d_particles_sort
        self.d_grid_box_count = None

    def summarize(self):
        bsize = 16*16
        threads = (16, 16)
        nblocks = self.grid_dim*self.grid_dim / bsize
        nblocks = ceil(sqrt(nblocks))
        blocks = (nblocks, nblocks)
        #print("cumm ", self.d_grid_box_cumm.copy_to_host())

        g_summarize[blocks, threads](self.d_particles, self.d_grid_box_cumm, 
                                     self.grid_dim, self.d_COMs)

    def evaluate(self):
        # because of the limits of a block, we can't do one block per box, so let's spread
        # the boxes into the x axis, and use the y axis to have more than 1024 threads 

        # how many blocs we need to cover all particles
        yblocks = ceil(self.n_particles/THREADS_PER_BLOCK)
        #one block per box
        blocks = (self.grid_dim*self.grid_dim, yblocks)
        threads = min(THREADS_PER_BLOCK, self.n_particles)
        logging.debug(f"Running evaluate kernel with blocks: {blocks}   threads {threads}")

        g_evaluate_boxes[blocks, threads](self.d_particles, self.grid_dim, self.d_grid_box_cumm, self.d_COMs, self.G)

    def timestep(self):
        tick = float(Config.get("bh", "tick_seconds"))
        blocks = ceil(self.n_particles / THREADS_PER_BLOCK)
        threads = THREADS_PER_BLOCK
        g_tick_particles[blocks, threads](self.d_particles, tick)

    def ensure_particles_id_ordered(self):
        self.d_particles.copy_to_host(self.particles)
        self.particles.view(p.fieldsstr).sort(order=p.idf, axis=0)

    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i].copy()
            return samples
