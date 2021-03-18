import logging
import numpy as np
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy.lib.recfunctions import structured_to_unstructured as unst
from math import sqrt, ceil
from .singlegpu import SingleGPUBarnesHut
from numba import cuda

#import kernels
from barneshut.kernels.grid_decomposition.singlegpu.grid import *
from barneshut.kernels.grid_decomposition.multigpu.grid import *

LEAF_OCCUPANCY = 0.7
# TODO: improve this thing to be an optimal
THREADS_PER_BLOCK = 128

# Inherit single gpu since we have some common stuff
class MultiGPUBarnesHut (SingleGPUBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.ngpus = int(Config.get("multigpu", "ngpus"))
        self.device_arrays_initd = False
    
        self.gpu_cells = {}


    class GPUCell:
        def __init__(self, resident_gpu, host_particles, grid_dim):
            self.resident_gpu = resident_gpu
            self.grid_dim = grid_dim
            self.host_particles = host_particles

            self.particle_start = None
            self.particle_end = None
            self.debug = logging.root.level == logging.DEBUG

        def requires_cuda_ctx(function):
            """Decorator that automatically activates the correct cuda context
            for this object.
            """
            def requires_cuda_ctx_wrapper(*args):
                self = args[0]
                with cuda.gpus[self.resident_gpu]:
                    if self.debug:
                        dvc = cuda.get_current_device()
                        print(f"Running in cuda context, GPU #{dvc.id}")
                    function(*args)
            return requires_cuda_ctx_wrapper

        def alloc_on_device(self):
            self.d_grid_box_count = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_grid_box_cumm  = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_COMs           = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float64)
            self.d_particles      = cuda.to_device(unst(self.particles))

        def set_particle_range(self, start, end):
            self.start = start
            self.end = end
            # TODO: perhaps this already copies into gpu

        @requires_cuda_ctx
        def place_particles_copy_back(self, min_xy, step):
            """Place particles in the grid, and copy them back to the
            host particles array
            """
            g_place_particles(particles, particles_ordered, min_xy, step, grid_dim, grid_box_count, grid_box_cumm):
            #TODO

        @requires_cuda_ctx
        def summarize_and_collect(self, host_COMs):
            # call COM kernel, get result back and put each into
            # host_COMS
            pass

        @requires_cuda_ctx
        def broadcast_COMs(self, host_COMS):
            # copy host COMs into the device
            pass


    def __init_gpu_cells(self):
        #TODO
        pass
        
    def call_method_all_cells(self, fn_name, *args):
        # TODO: this needs to be done by threads since numba calls are blocking
        # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
        # https://stackoverflow.com/questions/31159165/python-threadpoolexecutor-on-method-of-instance
        for cell in self.gpu_cells.values():
            getattr(cell, fn_name)(*args)

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.__create_grid()
        # we now have access to: self.grid_dim, self.step and self.min_xy
        
        #initialize (once) our cells, one per GPU
        self.__init_gpu_cells()

        #split particle array in equal parts
        for cell in self.gpu_cells.values():
            cell.set_particle_range(0,0) # TODO

        # tell every cell to place the particle range they have into
        # the grid
        self.call_method_all_cells("place_particles_copy_back")
        cuda.synchronize()

        # sort host particles array, marking ranges of each group of cells
        # there is a way of doing this without a sort, just gotta figure it out
        for cell in self.gpu_cells.values():
            cell.set_particle_range(0,0) # TODO
        cuda.synchronize()
        
    def summarize(self):
        self.host_COMs = np.zeros((self.grid_dim,self.grid_dim, 3))
        self.call_method_all_cells("summarize_and_collect", self.host_COMs)
        cuda.synchronize()
        self.call_method_all_cells("broadcast_COMs", self.host_COMs)
        cuda.synchronize()

    def evaluate(self):
        




    def get_particles(self, sample_indices=None):
        # if sample_indices is None:
        #     return self.particles
        # else:
        #     samples = {}
        #     for i in sample_indices:
        #         samples[i] = self.particles[i].copy()
        #     return samples
        pass

    def timestep(self):
       pass