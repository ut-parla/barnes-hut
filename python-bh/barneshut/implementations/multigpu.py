import logging
import numpy as np
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square, get_neighbor_cells
from math import sqrt, ceil
from numba import cuda
import threading, queue
from .base import BaseBarnesHut
from functools import wraps
from itertools import product
import barneshut.internals.particle as p

#import kernels
from barneshut.kernels.grid_decomposition.gpu.grid import *

MAX_X_BLOCKS = 65535
#max is 65535

class MultiGPUBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None
        self.ngpus = int(Config.get("multigpu", "ngpus"))
        self.device_arrays_initd = False
        self.gpu_cells_initd = False
        self.gpu_cells = {}
        logging.debug(f"GPUs available: {cuda.gpus}")

    class GPUCell:
        def __init__(self, resident_gpu):
            self.resident_gpu = resident_gpu
            self.debug = logging.root.level == logging.DEBUG
            self.queue = queue.Queue()
            self.done_event = threading.Event()
            self.thread = None
            self.G = float(Config.get("bh", "grav_constant"))
            self.keep_running = True
            self.threads_per_block = int(Config.get("cuda", "threads_per_block"))

        def launch(self):
            """Launch ourselves as thread"""
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

        def run(self):
            """Loop listening to the queue. Incoming messages from the 
            queue have function name to execute and optional args.
            Everything is run on the threads' GPU.
            """
            with cuda.gpus[self.resident_gpu]:
                logging.debug(f"Thread running in cuda context, GPU #{self.resident_gpu}")
                while self.keep_running:
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
            logging.debug(f"GPU {self.resident_gpu}: thread stopping...")

        def notify_when_done(fn):     
            @wraps(fn)
            def wrapper(*args, **kwargs):
                fn(*args, **kwargs)
                args[0].done_event.set()
            return wrapper    

        #self.zz = np.zeros((self.grid_dim,self.grid_dim), dtype=np.int)
        #self.d_grid_box_count = cuda.to_device(self.zz)
        #self.d_COMs           = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float64)

        def set_grid_boxes(self, grid_boxes):
            self.grid_boxes = grid_boxes

        def set_particle_range(self, particles, start, end):
            self.particles = particles
            self.start = start
            self.end = end

        @notify_when_done
        def terminate(self):
            self.keep_running = False

        @notify_when_done
        def copy_in_place_particles_copy_out(self, host_grid_count, lock):
            """Place particles in the grid, and copy them back to the
            host particles array
            """
            self.d_particles = cuda.to_device(self.particles[self.start:self.end])

            self.zz = np.zeros((self.grid_dim,self.grid_dim), dtype=np.int)
            self.d_grid_box_count = cuda.to_device(self.zz)

            #run kernel
            blocks = ceil((self.end-self.start) / self.threads_per_block)
            blocks = min(blocks, MAX_X_BLOCKS)
            threads = self.threads_per_block
            logging.debug(f"launching {blocks} {threads} kernels")
            g_place_particles[blocks, threads](self.d_particles, self.min_xy, self.step,
                          self.grid_dim, self.d_grid_box_count)
            cuda.synchronize()
            self.d_particles.copy_to_host(self.particles[self.start:self.end])
            with lock:
                host_grid_count += self.d_grid_box_count.copy_to_host()

        @notify_when_done
        def copy_particle_range_to_device(self, host_particles):
            try:
                self.d_particles.copy_to_device(host_particles[self.start:self.end])
                logging.debug(f"GPU {self.resident_gpu}: copied our slice directly to preallocated darray")
            except:
                logging.debug(f"GPU {self.resident_gpu}: couldn't copy directly, need to realocate")
                self.d_particles = cuda.to_device(host_particles[self.start:self.end])

        @notify_when_done
        def summarize_and_collect(self, host_COMs, lock, grid_ranges):
            # Before we had a subset of unordered points. now we have a subset of ordered points
            # so we need to recalculate our cumm box count before summarizing
            blocks = ceil(len(self.grid_boxes) / self.threads_per_block)
            threads = min(self.threads_per_block, len(self.grid_boxes))
            
            fb_x, fb_y = self.grid_boxes[0]
            offset = grid_ranges[fb_x, fb_y, 0]

            self.d_COMs = cuda.device_array((self.grid_dim,self.grid_dim, 3), dtype=np.float64)
            g_summarize_w_ranges[blocks, threads](self.d_particles, self.grid_boxes, offset, grid_ranges, 
                self.grid_dim, self.d_COMs)
            
            cuda.synchronize()
            h_COMs = self.d_COMs.copy_to_host()
            with lock:
                host_COMs += h_COMs

        @notify_when_done
        def store_COMs(self, host_COMS):
            self.d_COMs.copy_to_device(host_COMS)

        def get_close_neighbors(self):
            cn = set()
            logging.debug(f"GPU {self.resident_gpu}: boxes {self.grid_boxes}")
            for box in self.grid_boxes:
                cn |= set(get_neighbor_cells(tuple(box), self.grid_dim))
            cn -= set([(x,y) for x,y in self.grid_boxes])
            logging.debug(f"GPU {self.resident_gpu}: close neighbors {cn}")
            return cn

        @notify_when_done
        def copy_nonresident_neighbors(self, grid_ranges):
            """ Because the grid is split across multiple GPUs,
            some of the neighbors' particles of this GPU's boxes will not
            reside in this GPU but they are required for p2p kernel
            calculation. We must find which neighbors are these,
            copy them to our GPU and also copy an index so that
            the CUDA kernel can find them.
            """
            cn = self.get_close_neighbors()
            total_neighbor_particles = 0
            for neighbor in cn:
                x, y = neighbor
                start, end = grid_ranges[x,y]
                total_neighbor_particles += end-start
            logging.debug(f"GPU {self.resident_gpu}: close neighbors total particles is {total_neighbor_particles}.")

            #save some space by just allocating px, py and mass
            neighbor_particles = np.empty((total_neighbor_particles,3))
            indices = np.zeros((self.grid_dim,self.grid_dim, 2), dtype=np.int32)

            idx = 0
            for x,y in cn:
                start, end = grid_ranges[x,y]
                total = end-start
                neighbor_particles[idx:idx+total] = self.particles[start:end, 0:3]
                indices[x,y] = idx, idx+total
                idx += total

            self.d_neighbors = cuda.to_device(neighbor_particles)
            self.d_neighbors_indices = cuda.to_device(indices)

        @notify_when_done
        def evaluate(self, grid_ranges):
            # because of the limits of a block, we can't do one block per box, so let's spread
            # the boxes into the x axis, and use the y axis to have more than 1024 threads 
            d_cells = cuda.to_device(self.grid_boxes)

            pblocks = ceil((self.end-self.start)/self.threads_per_block)
            pblocks = min(pblocks, MAX_X_BLOCKS)
            gblocks = min(len(d_cells), MAX_X_BLOCKS)

            blocks = (pblocks, gblocks)
            threads = self.threads_per_block
            offset = self.start

            print(f"Running evaluate kernel with {len(d_cells)} boxes, blocks: {blocks}   threads {threads}")
            g_evaluate_parla_multigpu[blocks, threads](self.d_particles, self.grid_boxes, grid_ranges, offset, self.grid_dim, 
                self.d_COMs, self.d_neighbors_indices, self.d_neighbors, self.G)
            cuda.synchronize()

        @notify_when_done
        def copy_into_host_particles(self, host_particles):
            x = self.d_particles.copy_to_host()
            host_particles[self.start:self.end, :] = x

        @notify_when_done
        def tick_particles(self):
            blocks = ceil(self.end-self.start / self.threads_per_block)
            threads = self.threads_per_block
            tick = float(Config.get("bh", "tick_seconds"))
            g_tick_particles[blocks, threads](self.d_particles, tick)
            cuda.synchronize()
            
    def __init_gpu_cells(self):
        #create the threads if we haven't
        if not self.gpu_cells_initd:
            for i in range(self.ngpus):
                self.gpu_cells[i] = MultiGPUBarnesHut.GPUCell(i)
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
        self.set_particles_bounding_box() 
        self.__init_gpu_cells()

        #split particle array in equal parts
        logging.debug("Splitting particle array into chunks:")
        acc = 0
        for i, chunk in enumerate(np.array_split(self.particles, self.ngpus)):
            logging.debug(f"   chunk {i}:  {acc}-{acc+len(chunk)}")
            self.gpu_cells[i].set_particle_range(self.particles, acc, acc+len(chunk))
            acc += len(chunk)
        
        # tell every cell to place the particle range they have into the grid
        self.grid_count = np.zeros((self.grid_dim,self.grid_dim), dtype=np.int)
        self.grid_ranges = np.empty((self.grid_dim,self.grid_dim, 2), dtype=np.int)
        self.call_method_all_cells("copy_in_place_particles_copy_out", self.grid_count, threading.Lock())
        self.particles.view(p.fieldsstr).sort(order=[p.gxf, p.gyf], axis=0, kind="stable")

        acc = 0
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                self.grid_ranges[i,j] = acc, acc+self.grid_count[i,j]
                acc += self.grid_count[i,j]

        all_boxes = []
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                all_boxes.append((i,j))

        for i, box_range in enumerate(np.array_split(all_boxes, self.ngpus)):
            fb_x, fb_y = box_range[0]
            lb_x, lb_y = box_range[-1]
            start = self.grid_ranges[fb_x, fb_y, 0]
            end = self.grid_ranges[lb_x, lb_y, 1]
            logging.debug(f"slice of gpu {i}: {start}-{end}")
            self.gpu_cells[i].set_particle_range(self.particles, start, end)
            self.gpu_cells[i].set_grid_boxes(box_range)

        self.call_method_all_cells("copy_particle_range_to_device", self.particles)

    def summarize(self):
        """ Every GPU pretty much needs all COMs,
        so calculate them and broadcast
        """
        # host_COMs is the actual px, py, mass of each COM
        self.host_COMs = np.zeros((self.grid_dim,self.grid_dim, 3), dtype=np.float)
        self.call_method_all_cells("summarize_and_collect", self.host_COMs, 
                    threading.Lock(), self.grid_ranges)
        self.call_method_all_cells("store_COMs", self.host_COMs)
        logging.debug(f"Host COMs: {self.host_COMs}")

    def evaluate(self):
        self.call_method_all_cells("copy_nonresident_neighbors", self.grid_ranges)
        self.call_method_all_cells("evaluate", self.grid_ranges)

    def get_particles(self, sample_indices=None):
        if sample_indices is None:
            return self.particles
        else:
            samples = {}
            for i in sample_indices:
                samples[i] = self.particles[i].copy()
            return samples

    def timestep(self):
        self.call_method_all_cells("tick_particles")

    def ensure_particles_id_ordered(self):
        self.call_method_all_cells("copy_into_host_particles", self.particles)
        self.particles.view(p.fieldsstr).sort(order=p.idf, axis=0)

    def cleanup(self):
        super().cleanup()
        logging.debug("Terminating threads...")
        self.call_method_all_cells("terminate")
        for cell in self.gpu_cells.values():
            cell.thread.join()
        
        self.device_arrays_initd = False
        self.gpu_cells_initd = False
        self.gpu_cells = {}
        self.grid = None
        logging.debug("done!")
