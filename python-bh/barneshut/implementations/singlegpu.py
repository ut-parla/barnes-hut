import logging
import numpy as np
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy.lib.recfunctions import structured_to_unstructured as unst
from math import sqrt, ceil
from .base import BaseBarnesHut
from numba import cuda

#import kernels
from barneshut.kernels.grid_decomposition.singlegpu.grid import *

LEAF_OCCUPANCY = 0.7

# TODO: improve this thing to be an optimal
THREADS_PER_BLOCK = 128

class SingleGPUBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
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
        if not self.device_arrays_initd:
            # alloc gpu arrays
            self.d_grid_box_count = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_grid_box_cumm  = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_particles      = cuda.to_device(unst(self.particles))
            self.device_arrays_initd = True

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.__create_grid()        
        self.__init_device_arrays()

        print("shape: ", unst(self.particles).shape) 

        self.d_particles_sort = cuda.device_array_like(unst(self.particles))

        # alloc one int so we know the biggest box
        d_max_box = cuda.device_array(1, dtype=np.int)

        #call kernel
        blocks = ceil(self.n_particles / THREADS_PER_BLOCK)
        threads = THREADS_PER_BLOCK
        g_place_particles[blocks, threads](self.d_particles, self.d_particles_sort, self.min_xy, self.step,
                          self.grid_dim, self.d_grid_box_count, self.d_grid_box_cumm, d_max_box)

        cuda.synchronize()

        #get biggest box and sabe
        self.max_box = d_max_box.copy_to_host()[0]
        
        # Swap so original d_particles is deallocated. dealloc d_grid_box_count
        self.d_particles = self.d_particles_sort
        self.d_grid_box_count = None

        if self.debug:
            parts = self.d_particles_sort.copy_to_host()
            logging.debug("Got particles from device, here are the first 10:")
            for i in range(10):
                logging.debug(f"    {parts[i]}")

    def summarize(self):
        # alloc
        self.d_COMs = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float)
        
        blocks = 1
        threads = (self.grid_dim, self.grid_dim)
        g_summarize[blocks, threads](self.d_particles, self.d_grid_box_cumm, 
                                     self.grid_dim, self.d_COMs)
        cuda.synchronize()

    def evaluate(self):
        #fix blocks/threads. we need to find maximum particles a box has and use that
        blocks = (self.grid_dim, self.grid_dim)
        threads = 100

        g_evaluate_boxes[blocks, threads](self.d_particles, self.grid_dim, self.d_grid_box_cumm, self.d_COMs, self.G)
        cuda.synchronize()

    def timestep(self):
       pass