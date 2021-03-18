import logging
import numpy as np
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square
from numpy.lib.recfunctions import structured_to_unstructured as unst
from math import sqrt, ceil
from numba import cuda
import threading, queue
from .base import BaseBarnesHut
from functools import wraps
from itertools import product

#import kernels
from barneshut.kernels.grid_decomposition.singlegpu.grid import *
from barneshut.kernels.grid_decomposition.multigpu.grid import *

LEAF_OCCUPANCY = 0.7
# TODO: improve this thing to be an optimal
THREADS_PER_BLOCK = 128

#
# TODO: Need to find a way of having a thread per GPU in a way that doesn't
# make them possibly switch gpu context to a different one across calls.
# a.k.a. each thread is pinned to one context. 
# answer is probably queues and threads listening
#

# Inherit single gpu since we have some common stuff
class MultiGPUBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
        self.grid = None
        self.ngpus = int(Config.get("multigpu", "ngpus"))
        self.device_arrays_initd = False
        self.gpu_cells_initd = False
        self.gpu_cells = {}

    class GPUCell:
        def __init__(self, resident_gpu, host_particles):
            self.resident_gpu = resident_gpu
            self.host_particles = unst(host_particles)
            self.debug = logging.root.level == logging.DEBUG
            self.queue = queue.Queue()
            self.done_event = threading.Event()
            self.particle_start = None
            self.particle_end = None
            self.thread = None
            self.device_allocd = False
            self.grid_dim = self.min_xy = self.step = None

        def launch(self):
            """Launch ourselves as thread"""
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

        def run(self):
            """Loop listening to the queue. Incoming messages from the 
            queue have function name to execute and optional args.
            Everything is run on the threads' GPU.
            """
            print(f"devices: {cuda.gpus}. resident: {self.resident_gpu}")
            with cuda.gpus[self.resident_gpu]:
                logging.debug(f"Thread running in cuda context, GPU #{self.resident_gpu}")
                while True:
                    # block until we get a fn to execute
                    msg = self.queue.get()
                    fn_name = msg[0]
                    # if len is 1, no args
                    if len(msg) == 1:
                        getattr(self, fn_name)()
                    # else, expand args
                    else:
                        args = msg[1]
                        getattr(self, fn_name)(*args)
        
        def notify_when_done(fn):     
            @wraps(fn)
            def wrapper(*args, **kwargs):
                fn(*args, **kwargs)
                args[0].done_event.set()
            return wrapper    

        def alloc_on_device(self):
            if not self.device_allocd:
                #TODO: check if grid_dim changed, because then we need to realloc
                self.d_grid_box_count = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
                self.d_grid_box_cumm  = cuda.device_array((self.grid_dim,self.grid_dim), dtype=np.int)
                self.d_COMs           = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float64)
                self.device_allocd = True

        def set_particle_range(self, start, end):
            self.start = start
            self.end = end

        @notify_when_done
        def copy_in_place_particles_copy_out(self):
            """Place particles in the grid, and copy them back to the
            host particles array
            """
            self.alloc_on_device()
            #copy in
            self.d_particles = cuda.to_device(self.host_particles[self.start:self.end])
            d_particles_sort = cuda.device_array_like(self.host_particles[self.start:self.end])
            
            #run kernel
            blocks = ceil(self.end-self.start / THREADS_PER_BLOCK)
            threads = THREADS_PER_BLOCK
            g_place_particles[blocks, threads](self.d_particles, d_particles_sort, self.min_xy, self.step, 
                                            self.grid_dim, self.d_grid_box_count, self.d_grid_box_cumm)
            
            #copy out
            self.d_particles = d_particles_sort
            self.d_particles.copy_to_host(self.host_particles[self.start:self.end])

        def copy_particle_range_to_device(self):
            try:
                self.d_particles.copy_to_device(self.host_particles[self.start:self.end])
                logging.debug(f"GPU {self.resident_gpu}: copied our slice directly to preallocated darray")
            except:
                logging.debug(f"GPU {self.resident_gpu}: couldn't copy directly, need to realocate")
                self.d_particles = cuda.to_device(self.host_particles[self.start:self.end])

        def summarize_and_collect(self, host_COMs):
            # call COM kernel, get result back and put each into
            # host_COMS
            pass

        def broadcast_COMs(self, host_COMS):
            # copy host COMs into the device
            pass

        def copy_nonresident_neighbors(self, grid_rages):
            """ Because the grid is split across multiple GPUs,
            some of the neighbors' particles of this GPU's boxes will not
            reside in this GPU but they are required for p2p kernel
            calculation. We must find which neighbors are these,
            copy them to our GPU and also copy an index so that
            the CUDA kernel can find them.
            """
            pass

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


    def __init_gpu_cells(self):
        #create the threads if we haven't
        if not self.gpu_cells_initd:
            for i in range(self.ngpus):
                self.gpu_cells[i] = MultiGPUBarnesHut.GPUCell(i, self.particles)
                self.gpu_cells[i].launch()
            self.gpu_cells_initd = True

        for cell in self.gpu_cells.values():
            cell.grid_dim = self.grid_dim
            cell.min_xy = self.min_xy
            cell.step = self.step

    def call_method_all_cells(self, fn_name, *args):
        #send message to all threads
        for cell in self.gpu_cells.values():
            cell.queue.put((fn_name, args))

        for cell in self.gpu_cells.values():
            cell.done_event.wait()
            cell.done_event.clear()

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.__create_grid()
        # we now have access to: self.grid_dim, self.step and self.min_xy
        
        #initialize (once) our cells, one per GPU
        self.__init_gpu_cells()

        #split particle array in equal parts
        logging.debug("Splitting particle array into chunks:")
        acc = 0
        for i, chunk in enumerate(np.array_split(self.particles, self.ngpus)):
            logging.debug(f"   chunk {i}:  {acc}-{acc+len(chunk)}")
            self.gpu_cells[i].set_particle_range(acc, acc+len(chunk))
            acc += len(chunk)
        
        # tell every cell to place the particle range they have into the grid
        self.call_method_all_cells("copy_in_place_particles_copy_out")

        # sort host particles array, marking ranges of each group of cells
        # there is a way of doing this without a sort, just gotta figure it out.
        # because we are scanning, might as well have ranges for each grid
        # TODO: something better
        self.particles.sort(order=('gx', 'gy'), axis=0)
        logging.debug(f"host particles after place/sort:\n{self.particles}")

        # map every box in the grid to a slice of the particles array since it is sorted
        self.grid_ranges = np.empty((self.grid_dim,self.grid_dim, 2), dtype=np.int)
        prev_box = self.particles[0][['gx', 'gy']]
        start = 0
        cnt = 0
        for i in range(len(self.particles)):
            current_box = self.particles[i][['gx', 'gy']]
            #print(f"current: {current_box}   prev: {prev_box}")
            #print(f"start  {start}   end  {start+cnt}")
            if current_box != prev_box or i == len(self.particles)-1:
                x, y = int(prev_box[0]), int(prev_box[1])
                if i == len(self.particles)-1:
                    cnt += 1
                self.grid_ranges[x,y] = start, start+cnt
                prev_box = current_box
                start +=cnt
                cnt = 0
            cnt += 1

        logging.debug(f"grid range: {self.grid_ranges}")

        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))
        for i, box_range in enumerate(np.array_split(all_boxes, self.ngpus)):
            start_box = box_range[0]
            end_box = box_range[-1]
            x,y = start_box
            start_p = self.grid_ranges[x,y][0]
            x,y = end_box
            end_p = self.grid_ranges[x,y][1]
            logging.debug(f"slice of gpu {i}: {start_p}-{end_p}")
            self.gpu_cells[i].set_particle_range(start_p, end_p)

        self.call_method_all_cells("copy_particle_range_to_device")

        import sys
        sys.exit(0)

    def summarize(self):
        """ Every GPU pretty much needs all COMs,
        so calculate them and broadcast
        """
        # host_COMs is the actual px, py, mass of each COM
        self.host_COMs = np.zeros((self.grid_dim,self.grid_dim, 3), dtype=np.float)
        self.call_method_all_cells("summarize_and_collect", self.host_COMs)
        self.call_method_all_cells("broadcast_COMs", self.host_COMs)

    def evaluate(self):
        self.call_method_all_cells("copy_nonresident_neighbors", self.grid_ranges)
        self.call_method_all_cells("evaluate", self.grid_ranges)

    def get_particles(self, sample_indices=None):
        pass

    def timestep(self):
       pass