import numpy as np
from parla import Parla
from parla.cpu import *
from parla.tasks import *
from .base import BaseBarnesHut
import barneshut.internals.particle as p
from barneshut.internals.config import Config
from barneshut.grid_decomposition import Box
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square, get_neighbor_cells,remove_bottom_left_neighbors
import logging
from timer import Timer
from math import ceil
from itertools import product


class ParlaBarnesHut (BaseBarnesHut):

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.grid = None
 
        from barneshut.kernels.gravity import get_gravity_kernel
        self.grav_kernel = get_gravity_kernel()

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
            Timer.print()
            self.cleanup()

    async def parla_place_particles(self):
        ppt = int(Config.get("parla_cpu_only", "placement_particles_per_task"))
        slices = ceil(self.n_particles / ppt)
        logging.debug(f"Launching {slices} parla tasks to calculate particle placement.")

        #print("before ", self.particles)
        placement_TS = TaskSpace("particle_placement")
        for i, pslice in enumerate(np.array_split(self.particles, slices)):
            @spawn(placement_TS[i])
            def particle_placement_task():
                pslice[:, p.gx:p.gy+1] = pslice[:, p.px:p.py+1]
                pslice[:, p.gx:p.gy+1] = (pslice[:, p.gx:p.gy+1] - self.min_xy) / self.step
                pslice[:, p.gx:p.gy+1] = np.clip(np.floor(pslice[:, p.gx:p.gy+1]), 0, self.grid_dim-1)

        await placement_TS

        self.particles.view(p.fieldsstr).sort(order=(p.gxf, p.gyf), axis=0, kind="stable")
    
    async def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        self.set_particles_bounding_box()
        self.create_grid_boxes()

        await self.parla_place_particles()

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

    async def summarize(self):
        bpt = int(Config.get("parla_cpu_only", "summarize_boxes_per_task"))
        slices = ceil((self.grid_dim * self.grid_dim) / bpt)
        logging.debug(f"Launching {slices} parla tasks to summarize.")
        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))
        summarize_TS = TaskSpace("summarize")

        for i, box_range in enumerate(np.array_split(all_boxes, slices)):
            @spawn(summarize_TS[i])
            def summarize_task():
                for box_xy in box_range:
                    x,y = box_xy
                    self.grid[x][y].get_COM()

        await summarize_TS

    async def evaluate(self):
        bpt = int(Config.get("parla_cpu_only", "eval_boxes_per_task"))
        slices = ceil((self.grid_dim * self.grid_dim) / bpt)
        logging.debug(f"Launching {slices} parla tasks to evaluate.")
        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))
        eval_TS = TaskSpace("evaluate")

        #boxes are inbalanced, so lets shuffle them
        np.random.shuffle(all_boxes)

        for i, box_range in enumerate(np.array_split(all_boxes, slices)):
            @spawn(eval_TS[i])
            def evaluate_task():
                for box_xy in box_range:
                    x,y = box_xy
                    box = self.grid[x][y]
                    if box.cloud.is_empty():
                        continue

                    neighbors = get_neighbor_cells(tuple(box_xy), self.grid_dim)
                    com_cells = []
                    for otherbox_xy in product(range(self.grid_dim), range(self.grid_dim)):
                        ox, oy = otherbox_xy
                        if self.grid[ox][oy].cloud.is_empty() or (x == ox and y == oy):
                            continue
                        if otherbox_xy not in neighbors:
                            com_cells.append(self.grid[ox][oy])

                    coms = Box.from_list_of_boxes(com_cells, is_COMs=True)
                    box.apply_force(coms, update_other=False)

                    # remove boxes that already computed their force to us (this function modifies neighbors list)
                    for otherbox_xy in remove_bottom_left_neighbors(box_xy, neighbors):
                        ox, oy = otherbox_xy
                        # if box is empty, just skip it
                        if self.grid[ox][oy].cloud.is_empty():
                            continue
                        box.apply_force(self.grid[ox][oy], update_other=True)

                    # we also need to interact with ourself
                    box.apply_force(box, update_other=False)

        await eval_TS
        
    async def timestep(self):
        bpt = int(Config.get("parla_cpu_only", "timestep_boxes_per_task"))
        slices = self.grid_dim * self.grid_dim
        logging.debug(f"Launching {slices} parla tasks to tick timestep.")
        all_boxes = list(product(range(self.grid_dim), range(self.grid_dim)))
        timestep_TS = TaskSpace("timestep")

        for i, box_range in enumerate(np.array_split(all_boxes, slices)):
            @spawn(timestep_TS[i])
            def timestep_task():
                for box in box_range:
                    x,y = box
                    self.grid[x][y].tick()

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