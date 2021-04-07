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


class ParlaBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None
        self.grid_cumm = None
        self.ngpus = int(Config.get("parla", "gpus_available"))

    def run(self,  check_accuracy=False):
        with Parla():
            @spawn()
            async def main():
                await self.run_bh(check_accuracy)

    async def run_bh(self, check_accuracy=False):
        """This sucks.. because everything is async in Parla and needs to be awaited,
        we need to copy/paste this method from base.py"""
        n_iterations = int(Config.get("general", "rounds"))
        self.checking_accuracy = check_accuracy
        if self.checking_accuracy:
            sample_indices = self.generate_sample_indices()
        with Timer.get_handle("end-to-end"):
            for _ in range(n_iterations):
                if self.checking_accuracy:
                    nsquared_sample = self.preround_accuracy_check(sample_indices)
                with Timer.get_handle("grid_creation"):
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
        #Timer.print()
        self.cleanup()

    async def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()

        cpu_tasks = int(Config.get("parla", "placement_cpu_tasks"))
        gpu_tasks = int(Config.get("parla", "placement_gpu_tasks"))
        total_tasks = cpu_tasks + gpu_tasks        
        logging.debug(f"Launching {cpu_tasks} cpu and {gpu_tasks} gpu tasks to calculate particle placement.")

        self.grid_cumm = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int32)
        grid_cumms = np.zeros((total_tasks, self.grid_dim, self.grid_dim), dtype=np.int32)
        placement_TS = TaskSpace("particle_placement")
        
        placements = []
        for _ in range(cpu_tasks):
            placements.append(cpu)
        for i in range(gpu_tasks):
            placements.append(gpu(i%self.ngpus))
            #placements.append(gpu)

        for i, pslice in enumerate(np.array_split(self.particles, total_tasks)):
            # approximate mem usage
            #memusage = pslice.nbytes * 1.1 if placements[i] is not cpu else 0
            #@spawn(placement_TS[i], placement=placements[i], memory=memusage)
            @spawn(placement_TS[i], placement=placements[i])
            def particle_placement_task():
                #particles_here = clone_here(pslice)
                particles_here = pslice
                cumm = grid_cumms[i]
                p_place_particles(particles_here, cumm, self.min_xy, self.grid_dim, self.step)
                if placements[i] is not cpu:
                    cuda.synchronize()
                #copy(pslice, particles_here)

        #await placement_TS
        post_placement_TS = TaskSpace("post_placement")
        #with Timer.get_handle("sort"):
        # for some reason using stable here is really expensive, maybe it's not using radix
        @spawn(post_placement_TS[0], [placement_TS])
        def sort_grid_task():
            self.particles.view(p.fieldsstr).sort(order=[p.gxf, p.gyf])  #, axis=0, kind="stable")
            
        self.grid_ranges = np.zeros((self.grid_dim, self.grid_dim, 2), dtype=np.int32)
        @spawn(post_placement_TS[1], [placement_TS])
        def acc_cumm_grid_task():
            # accumulate all cumm grids
            for i in range(total_tasks):
                self.grid_cumm += grid_cumms[i]
            acc = 0
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    self.grid_ranges[i,j] = acc, acc+self.grid_cumm[i,j]
                    acc += self.grid_cumm[i,j]

        await post_placement_TS

    async def summarize(self):
        cpu_tasks = int(Config.get("parla", "summarize_cpu_tasks"))
        gpu_tasks = int(Config.get("parla", "summarize_gpu_tasks"))
        total_tasks = cpu_tasks + gpu_tasks        
        logging.debug(f"Launching {cpu_tasks} cpu and {gpu_tasks} gpu tasks to summarize.")

        placements = []
        for _ in range(cpu_tasks):
            placements.append(cpu)
        for i in range(gpu_tasks):
            placements.append(gpu(i%self.ngpus))
            #placements.append(gpu)

        all_boxes = []
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                all_boxes.append((i,j))

        self.COMs = np.zeros((self.grid_dim, self.grid_dim, 3), dtype=np.float32)
        tasks_COMs = np.zeros((total_tasks, self.grid_dim, self.grid_dim, 3), dtype=np.float32)

        summarize_TS = TaskSpace("summarize")
        # because particles are sorted, and all_boxes is also sorted indices
        # we can assume that a subset of boxes here is contiguous
        for i, box_range in enumerate(np.array_split(all_boxes, total_tasks)):
            #memusage = self.particles[start:end].nbytes * 1.4 if placements[i] is not cpu else 0
            #@spawn(summarize_TS[i], placement=placements[i], memory=memusage)
            @spawn(summarize_TS[i], placement=placements[i])
            def summarize_task():
                fb_x, fb_y = box_range[0]
                lb_x, lb_y = box_range[-1]
                start = self.grid_ranges[fb_x, fb_y, 0]
                end = self.grid_ranges[lb_x, lb_y, 1]   
                particle_slice = self.particles[start:end]
                p_summarize_boxes(particle_slice, box_range, start, self.grid_ranges, self.grid_dim, tasks_COMs[i])
                if placements[i] is not cpu:
                    cuda.synchronize()
        await summarize_TS
        # accumulate COMs
        for i in range(total_tasks):
            self.COMs += tasks_COMs[i]

    async def evaluate(self):
        cpu_tasks = int(Config.get("parla", "evaluation_cpu_tasks"))
        gpu_tasks = int(Config.get("parla", "evaluation_gpu_tasks"))
        total_tasks = cpu_tasks + gpu_tasks        
        logging.debug(f"Launching {cpu_tasks} cpu and {gpu_tasks} gpu tasks to evaluate.")

        placements = []
        for _ in range(cpu_tasks):
            placements.append(cpu)
        for i in range(gpu_tasks):
            placements.append(gpu(i%self.ngpus))
            #placements.append(gpu)

        all_boxes = []
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                all_boxes.append((i,j))

        grid = {}
        # if we are using the cpu, let's build Cloud objects
        if cpu_tasks > 0:
            for box in all_boxes:
                x, y = box
                start, end = self.grid_ranges[x, y]
                grid[(x,y)] = Cloud.from_slice(self.particles[start:end])

        G = float(Config.get("bh", "grav_constant"))
        eval_TS = TaskSpace("evaluate")
        for i, box_range in enumerate(np.array_split(all_boxes, total_tasks)):
            #approximate 140% of particles
            #memusage = self.particles[start:end].nbytes * 2 if placements[i] is not cpu else 0
            #@spawn(eval_TS[i], placement=placements[i], memory=memusage)
            @spawn(eval_TS[i], placement=placements[i])
            def evaluate_task():
                fb_x, fb_y = box_range[0]
                lb_x, lb_y = box_range[-1]
                start = self.grid_ranges[fb_x, fb_y, 0]
                end = self.grid_ranges[lb_x, lb_y, 1]
                mod_particles = p_evaluate(self.particles, box_range, grid, self.grid_ranges, self.COMs, G, self.grid_dim)
                if placements[i] is not cpu:
                    copy(self.particles[start:end], mod_particles)
        await eval_TS

    async def timestep(self):
        cpu_tasks = int(Config.get("parla", "timestep_cpu_tasks"))
        gpu_tasks = int(Config.get("parla", "timestep_gpu_tasks"))
        total_tasks = cpu_tasks + gpu_tasks        
        logging.debug(f"Launching {cpu_tasks} cpu and {gpu_tasks} gpu tasks to timestep.")
        tick = float(Config.get("bh", "tick_seconds"))

        placements = []
        for _ in range(cpu_tasks):
            placements.append(cpu)
        for i in range(gpu_tasks):
            placements.append(gpu(i%self.ngpus))
            #placements.append(gpu)

        timestep_TS = TaskSpace("timestep")        
        for i, pslice in enumerate(np.array_split(self.particles, total_tasks)):
            #memusage = pslice.nbytes * 1.1 if placements[i] is not cpu else 0
            #@spawn(timestep_TS[i], placement=placements[i], memory=memusage)
            @spawn(timestep_TS[i], placement=placements[i])
            def timestep_task():
                particles_here = pslice
                p_timestep(particles_here, tick)

        await timestep_TS

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