import logging
import numpy as np
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy.lib.recfunctions import structured_to_unstructured as unst
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

    def __create_grid(self):
        """Figure out the grid dimension and points.
        This function sets:
        self.grid_dim
        self.step
        self.min_xy
        """
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
        self.step = (top_right[0] - bottom_left[0]) / self.grid_dim 
        self.min_xy = bottom_left
        
    def __init_device_arrays(self):
        """ Allocate arrays on the device and copy the particles
        array too. This is done only once.
        """
        if not self.device_arrays_initd:
            # alloc gpu arrays
            #self.d_grid_box_count = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_grid_box_cumm  = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int64)
            self.d_particles      = cuda.to_device(unst(self.particles))
            self.d_COMs           = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float)
            self.device_arrays_initd = True

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.__create_grid()        
        self.__init_device_arrays()

        # store the masses so we can unshuffle later, check timestep function.
        if self.checking_accuracy:
            self.ordered_masses = self.particles['mass']

        self.d_particles_sort = cuda.device_array_like(unst(self.particles))

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

        if self.debug:
            parts = self.d_particles_sort.copy_to_host()
            logging.debug("Got particles from device, here are the first 10:")
            for i in range(10):
                logging.debug(f"    {parts[i]}")

    def summarize(self):
        bsize = 16*16
        threads = (16, 16)
        nblocks = self.grid_dim*self.grid_dim / bsize
        nblocks = ceil(sqrt(nblocks))
        blocks = (nblocks, nblocks)
        g_summarize[blocks, threads](self.d_particles, self.d_grid_box_cumm, 
                                     self.grid_dim, self.d_COMs)

    def evaluate(self):
        # because of the limits of a block, we can't do one block per box, so let's spread
        # the boxes into the x axis, and use the y axis to have more than 1024 threads 
        yblocks = ceil(self.n_particles/THREADS_PER_BLOCK)
        blocks = (self.grid_dim*self.grid_dim, yblocks)
        threads = min(THREADS_PER_BLOCK, self.n_particles)
        logging.debug(f"Running evaluate kernel with blocks: {blocks}   threads {threads}")
        g_evaluate_boxes[blocks, threads](self.d_particles, self.grid_dim, self.d_grid_box_cumm, self.d_COMs, self.G)

    def timestep(self):
        tick = float(Config.get("bh", "tick_seconds"))
        blocks = ceil(self.n_particles / THREADS_PER_BLOCK)
        threads = THREADS_PER_BLOCK
        g_tick_particles[blocks, threads](self.d_particles, tick)

        # if checking accuracy, we need to copy it back to host
        if self.checking_accuracy:
            """Alright, fasten your seatbelts, this is some ugly
            research code.
            We first argsort the unshuffled particle masses, then argsort that
            array, which is needed to undo a sort, similar to what we do in the
            sequential check.
            Then, when we get the shuffled array from the GPU we argsort it
            and use the previous undo argsort to shuffle it back to the original
            positions. This assumes that masses are unique, which they probably aren't,
            so unless things are somehow stable, we might see different errors each run.
            """
            self.d_particles.copy_to_host(unst(self.particles))
            would_sort = np.argsort(self.ordered_masses)
            undo_sort = np.argsort(would_sort)
            would_sort_particles = np.argsort(self.particles, order=('mass'), axis=0)
            self.particles = self.particles[would_sort_particles][undo_sort]

    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i].copy()
            return samples
