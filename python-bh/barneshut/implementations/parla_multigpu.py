import numpy as np
from math import ceil
from itertools import product
import logging
from .base import BaseBarnesHut
from barneshut.internals.config import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square, get_neighbor_cells,remove_bottom_left_neighbors
from barneshut.kernels.grid_decomposition.gpu.grid import *
from barneshut.kernels.gravity import get_gravity_kernel
import barneshut.internals.particle as p
from timer import Timer

from parla import Parla
from parla.array import copy, clone_here
from parla.cuda import *
from parla.cpu import *
from parla.tasks import *
from parla.function_decorators import *

from barneshut.kernels.grid_decomposition.parla import *

CPU = True

class ParlaMultiGPUBarnesHut (BaseBarnesHut):

    class NBodyTask:
        def __init__(self):
            self.particle_slice = None


    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None
        self.grid_cumm = None

    def run(self, n_iterations, partitions=None, print_particles=False, check_accuracy=False):
        with Parla():
            @spawn()
            async def main():
                await self.run_bh(n_iterations, partitions, print_particles, check_accuracy)

    async def run_bh(self, n_iterations, partitions=None, print_particles=False, check_accuracy=False):
        """This sucks.. because everything is async in Parla and needs to be awaited,
        we need to copy/paste this method from base.py"""
        with Parla():
            self.checking_accuracy = check_accuracy
            if self.checking_accuracy:
                sample_indices = self.generate_sample_indices()
            with Timer.get_handle("end-to-end"):
                for _ in range(n_iterations):
                    if self.checking_accuracy:
                        nsquared_sample = self.preround_accuracy_check(sample_indices)
                    with Timer.get_handle("tree-creation"):
                        await self.create_tree()
                    with Timer.get_handle("summarization"):
                        await self.summarize()
                    for _ in range(self.evaluation_rounds):
                        with Timer.get_handle("evaluation"):
                            await self.evaluate()
                    if not self.skip_timestep:
                        with Timer.get_handle("timestep"):
                            await self.timestep()
                    if self.checking_accuracy:
                        self.ensure_particles_id_ordered()
                        self.check_accuracy(sample_indices, nsquared_sample)
            Timer.print()
            self.cleanup()

    async def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()
        self.grid_cumm = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int32)

        ppt = int(Config.get("parla", "placement_particles_per_task"))
        ntasks = ceil(self.n_particles / ppt)
        logging.debug(f"Launching {ntasks} parla tasks to calculate particle placement.")

        grid_cumms = np.zeros((ntasks, self.grid_dim, self.grid_dim), dtype=np.int32)
        print("before ", self.particles)
        placement_TS = TaskSpace("particle_placement")
        for i, pslice in enumerate(np.array_split(self.particles, ntasks)):
            @spawn(placement_TS[i], placement=cpu)
            def particle_placement_task():
                # ensure we have the particles and cumm grid
                particles_here = clone_here(pslice)
                cumm = clone_here(grid_cumms[i])
                # p_place_particles can be called on cpu or gpu
                p_place_particles(particles_here, cumm, self.min_xy, self.grid_dim, self.step)
                copy(grid_cumms[i], cumm)
                copy(pslice, particles_here)

        post_placement_TS = TaskSpace("post_placement")
        @spawn(post_placement_TS[0], [placement_TS])
        def sort_grid_task():
            self.particles.view(p.fieldsstr).sort(order=(p.gxf, p.gyf), axis=0)
            print("sorted ", self.particles)

        self.grid_ranges = np.zeros((self.grid_dim, self.grid_dim, 2), dtype=np.int32)
        @spawn(post_placement_TS[1], [placement_TS])
        def acc_cumm_grid_task():
            # accumulate all cumm grids
            for i in range(ntasks):
                self.grid_cumm += grid_cumms[i]
            acc = 0
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    self.grid_ranges[j, i] = acc, acc+self.grid_cumm[j, i]
                    acc += self.grid_cumm[j, i]
        await post_placement_TS

    async def summarize(self):
        bpt = int(Config.get("parla", "summarize_boxes_per_task"))
        ntasks = ceil((self.grid_dim * self.grid_dim) / bpt)
        logging.debug(f"Launching {ntasks} parla tasks to summarize.")
        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))

        self.COMs = np.zeros((self.grid_dim, self.grid_dim, 3), dtype=np.float32)
        tasks_COMs = np.zeros((ntasks, self.grid_dim, self.grid_dim, 3), dtype=np.float32)

        summarize_TS = TaskSpace("summarize")
        # because particles are sorted, and product also generates sorted indices
        # we can assume that a subset of boxes here is contiguous
        for i, box_range in enumerate(np.array_split(all_boxes, ntasks)):
            print("task ", i)
            @spawn(summarize_TS[i], placement=gpu(0))
            def summarize_task():
                fb_x, fb_y = box_range[0]
                lb_x, lb_y = box_range[-1]
                start = self.grid_ranges[fb_x, fb_y, 0]
                end = self.grid_ranges[lb_x, lb_y, 1]
                logging.debug(f"Summarize {i}  range {start}-{end}")                
                particle_slice = self.particles[start:end]

                p_summarize_boxes(particle_slice, box_range, self.grid_ranges, self.grid_dim, tasks_COMs[i])

        await summarize_TS
    
        # accumulate COMs
        for i in range(ntasks):
            self.COMs += tasks_COMs[i]
        print("task coms: ", self.COMs)

    async def evaluate(self):
        bpt = int(Config.get("parla", "eval_boxes_per_task"))
        ntasks = ceil((self.grid_dim * self.grid_dim) / bpt)
        logging.debug(f"Launching {ntasks} parla tasks to evaluate.")
        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))
        G = float(Config.get("bh", "grav_constant"))

        grid = {}
        if CPU:
            grav_kernel = get_gravity_kernel()
            for box in all_boxes:
                x, y = box
                start, end = self.grid_ranges[x, y]
                grid[(x,y)] = Cloud.from_slice(self.particles[start:end], grav_kernel)

        eval_TS = TaskSpace("evaluate")
        # # cpu tasks
        # for i, box_range in enumerate(np.array_split(all_boxes, ntasks)):
        #     @spawn(eval_TS[i])
        #     def evaluate_task():
        #         p_evaluate(grid, box_range, self.COMs, G, self.grid_dim)

        for i, box_range in enumerate(np.array_split(all_boxes, ntasks)):
            @spawn(eval_TS[i], placement=gpu(0))
            def evaluate_task():                
                # let's get all the neighbors we need
                p_evaluate(self.particles, box_range, grid, self.grid_ranges, self.COMs, G, self.grid_dim)

        await eval_TS
        
    async def timestep(self):
        pass

    def ensure_particles_id_ordered(self):
        # just sort by id since they were shuffled
        self.particles.view(p.fieldsstr).sort(order=p.idf, axis=0)

    def get_particles(self, sample_indices=None):
         if sample_indices is None:
             return self.particles
         else:
             samples = {}
             for i in sample_indices:
                 samples[i] = self.particles[i].copy()
             return samples