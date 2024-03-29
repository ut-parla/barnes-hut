import logging
from math import sqrt, pow, ceil
from itertools import combinations, product
import random
from typing import Tuple

import numpy as np
from numpy.lib import recfunctions as rfn
import pykokkos as pk

from .base import BaseBarnesHut
from barneshut.grid_decomposition import Box
from barneshut.internals.config import Config
from barneshut.internals.particle import Particle, particle_type
from barneshut.kernels.helpers import next_perfect_square, get_bounding_box
from timer import Timer

SAMPLE_SIZE = 10
LEAF_OCCUPANCY = 0.7
N_DIM = 2

@pk.functor
class PyKokkosConcatenatedBox:
    def __init__(self, box_1, box_2, use_COM):
        self_box = box_1 if box_1.n > box_2.n else box_2
        other_box = box_1 if box_1.n < box_2.n else box_2

        self.p_pos: pk.View2D[pk.double] = self_box.position
        self.p_mass: pk.View1D[pk.double] = self_box.mass
        self.p_accel: pk.View2D[pk.double] = self_box.acceleration
        self.cloud_pos: pk.View2D[pk.double] = other_box.COM_position if use_COM else other_box.position
        self.cloud_mass: pk.View1D[pk.double] = other_box.COM_mass if use_COM else other_box.mass
        self.cloud_accel: pk.View2D[pk.double] = other_box.COM_acceleration if use_COM else other_box.acceleration

        self.n: int = self.cloud_pos.extent(0)
        self.G: float = float(Config.get("bh", "grav_constant"))
        self.is_self_self: int = 1 if box_1 is box_2 else 0

    @pk.workunit
    def gravity(self, tid: int):
        for i in range(self.n):
            dif_1: float = self.p_pos[tid][0] - self.cloud_pos[i][0]
            dif_2: float = self.p_pos[tid][1] - self.cloud_pos[i][1]

            dist: float = sqrt(((dif_1 * dif_1) + (dif_2 * dif_2)))
            f: float = (self.G * self.p_mass[tid] * self.cloud_mass[i]) / (dist * dist * dist)

            self.p_accel[tid][0] -= f * dif_1 / self.p_mass[tid]
            self.p_accel[tid][1] -= f * dif_2 / self.p_mass[tid]

            if self.is_self_self != 0:
                self.cloud_accel[i][0] += f * dif_1 / self.cloud_mass[i]
                self.cloud_accel[i][1] += f * dif_2 / self.cloud_mass[i]

@pk.functor
class PyKokkosBox:
    def __init__(self, bottom_left: Tuple[int, int], top_right: Tuple[int, int]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.n = 0

        self.max_particles: int = int(Config.get("grid", "max_particles_per_box"))
        self.tick_parameter: float = float(Config.get("bh", "tick_seconds"))

        self.position: pk.View2D[pk.double] = pk.View([self.max_particles, N_DIM], pk.double)
        self.velocity: pk.View2D[pk.double] = pk.View([self.max_particles, N_DIM], pk.double)
        self.mass: pk.View1D[pk.double] = pk.View([self.max_particles], pk.double)
        self.acceleration: pk.View2D[pk.double] = pk.View([self.max_particles, N_DIM], pk.double)

        self.COM_initialized = False
        self.COM_position: pk.View2D[pk.double] = pk.View([2, N_DIM], pk.double)
        self.COM_velocity: pk.View2D[pk.double] = pk.View([2, N_DIM], pk.double)
        self.COM_mass: pk.View1D[pk.double] = pk.View([2], pk.double)
        self.COM_acceleration: pk.View2D[pk.double] = pk.View([2, N_DIM], pk.double)

    def is_empty(self) -> bool:
        return self.n == 0

    def is_full(self) -> bool:
        return self.n >= self.max_particles

    def add_particle(self, p: Particle) -> None:
        if self.is_full():
            print("adding to a full leaf, something is wrong")

        self.position[self.n][0] = p["px"]
        self.position[self.n][1] = p["py"]
        self.velocity[self.n][0] = p["vx"]
        self.velocity[self.n][1] = p["vy"]
        self.mass[self.n] = p["mass"]
        self.acceleration[self.n][0] = 0.0
        self.acceleration[self.n][1] = 0.0
        self.n += 1

        # TODO: this might work
        # self.position[self.n] = [p["px"], p["py"]]
        # self.velocity[self.n] = [p["vx"], p["vy"]]
        # self.mass[self.n] = p["mass"]
        # self.acceleration[self.n] = 0.0
        # self.n += 1

    def get_COM(self):
        if not self.COM_initialized:
            # equations taken from http://hyperphysics.phy-astr.gsu.edu/hbase/cm.html
            M = pk.parallel_reduce(self.n, self.COM_mass_kernel)
            x_avg = pk.parallel_reduce(self.n, self.COM_x_kernel) / M
            y_avg = pk.parallel_reduce(self.n, self.COM_y_kernel) / M
            self.init_COM((x_avg, y_avg), M)
            self.COM_initialized = True

    def init_COM(self, coords: Tuple[pk.double, pk.double], M: pk.double):
        self.COM_position[0][0] = coords[0]
        self.COM_position[0][1] = coords[1]
        self.COM_position[1][0] = 0
        self.COM_position[1][1] = 0
        self.COM_mass[0] = M

    def get_corners(self):
        x1, x2 = self.bottom_left[0], self.top_right[0]
        y1, y2 = self.bottom_left[1], self.top_right[1]
        return ((x1, x2), (y1, y2))

    def tick(self):
        pk.parallel_for(self.n, self.tick_kernel)

    @pk.workunit
    def COM_mass_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.mass[tid]
    
    @pk.workunit
    def COM_x_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.position[tid][0] * self.mass[tid]
        
    @pk.workunit
    def COM_y_kernel(self, tid: int, acc: pk.Acc[pk.double]):
        acc += self.position[tid][1] * self.mass[tid]

    def approximation_distance(self, other_box) -> bool:
        corners1 = self.get_corners()
        corners2 = other_box.get_corners()

        # there's gotta be a better way to do this
        for x in corners1[0]:
            for y in corners1[1]:
                px1, px2 = corners2[0]
                py1, py2 = corners2[1]
                if (
                    ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
                    ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
                   ):
                    return False
        for x in corners2[0]:
            for y in corners2[1]:
                px1, px2 = corners1[0]
                py1, py2 = corners1[1]
                if (
                    ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
                    ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
                   ):
                    return False

        return True

    def apply_force(self, other_box) -> None:
        if self.is_empty() or other_box.is_empty():
            return
        use_COM = self.approximation_distance(other_box)
        other_box.get_COM()

        concatenated = PyKokkosConcatenatedBox(self, other_box, use_COM)
        pk.parallel_for(concatenated.p_pos.extent(0), concatenated.gravity)

    @pk.workunit
    def tick_kernel(self, tid: int):
        self.velocity[tid][0] += self.acceleration[tid][0] * self.tick_parameter
        self.velocity[tid][1] += self.acceleration[tid][1] * self.tick_parameter
        self.acceleration[tid][0] = 0
        self.acceleration[tid][1] = 0
        self.position[tid][0] += self.velocity[tid][0] * self.tick_parameter
        self.position[tid][1] += self.velocity[tid][1] * self.tick_parameter

class PyKokkosBarnesHut(BaseBarnesHut):
    """ PyKokkos implementation of nbody."""

    def __init__(self):
        super().__init__()
        space: str = Config.get("pykokkos", "space")
        if space == "Cuda":
            pk.set_default_space(pk.Cuda)
        else:
            pk.set_default_space(pk.OpenMP)
        self.grid = None

    def read_particles_from_file(self, filename):
        """Call the base class method and construct the kernels object"""

        super().read_particles_from_file(filename)

    def __next_perfect_square(self, n):
        if n%n**0.5 != 0:
            return pow( ceil(sqrt(n))  , 2)
        return n

    def __get_bounding_box(self):
        """ Get bounding box coordinates around all particles.
        Returns the bottom left and top right corner coordinates, making
        sure that it is a square.
        """
        # https://numpy.org/doc/stable/user/basics.rec.html#indexing-and-assignment-to-structured-arrays
        parts = rfn.structured_to_unstructured(self.particles[['px', 'py']], copy=False)
        max_x, max_y = np.max(parts, axis=0)[:2]
        min_x, min_y = np.min(parts, axis=0)[:2]

        x_edge, y_edge = max_x - min_x, max_y - min_y 
        if x_edge >= y_edge:
            max_y += (x_edge - y_edge)
        else:
            max_x += (y_edge - x_edge)

        assert (max_x-min_x)==(max_y-min_y)
        return (min_x, min_y), (max_x, max_y)

    def __create_grid(self, bottom_left, top_right, grid_dim):
        # x and y have the same edge length, so get x length
        step = (top_right[0]-bottom_left[0]) / grid_dim
        # create grid as a matrix, starting from bottom left
        self.grid = []
        logging.debug(f"Grid: {bottom_left}, {top_right}")
        for i in range(grid_dim):
            row = []
            for j in range(grid_dim):
                x = bottom_left[0] + (i*step)
                y = bottom_left[1] + (j*step)
                row.append(PyKokkosBox((x,y), (x+step, y+step)))
                logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        # get bounding box around all particles
        unstr_points = rfn.structured_to_unstructured(self.particles[['px', 'py']], copy=False)
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
        grid_dim = int(sqrt(nleaves))

        logging.debug(f'''With {LEAF_OCCUPANCY} occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {grid_dim}.
                Grid will be {grid_dim}x{grid_dim}''')
        
        self.__create_grid(bottom_left, top_right, grid_dim)
        step =  (top_right[0] - bottom_left[0]) / grid_dim

        # placements is an array mapping points to their position in the matrix
        # this is just so we can easily map to numpy/cuda later
        points = rfn.structured_to_unstructured(self.particles[['px', 'py']], copy=True)

        # call kernel to place points
        with Timer.get_handle("placement_kernel"):
            points = (points - bottom_left) / step
            # truncate and convert to int
            points = np.trunc(points) #.astype(int, copy=False)
            points = np.clip(points, 0, grid_dim-1)

        with Timer.get_handle("grid assign"):
            for i in range(len(points)):
                x, y = points[i].astype(int, copy=False)
                logging.debug(f"adding point {i} ({self.particles[i]}) to box {x}/{y}")
                self.grid[x][y].add_particle(self.particles[i])

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):
        n = len(self.grid)
        # do all distinct pairs interaction
        cells = product(range(n), range(n))
        pairs = combinations(cells, 2)

        for p1, p2 in pairs:
            l1 = self.grid[p1[0]][p1[1]]
            l2 = self.grid[p2[0]][p2[1]]
            l1.apply_force(l2)

        # and all self to self interaction
        for l in range(n):
            leaf = self.grid[l][l]
            leaf.apply_force(leaf)

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].tick()

    def run(self, check_accuracy=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""

        with Timer.get_handle("end-to-end"):
            n_iterations = int(Config.get("general", "rounds"))
            for _ in range(n_iterations):
                # If we're checking accuracy, we need to save points before calculation
                if check_accuracy:
                    self.backup_particles()

                # Step 1: create tree (if Barnes-Hut), group-by points by box (if Decomposition)
                with Timer.get_handle("grid_creation"):
                    self.create_tree()

                # Step 2: summarize.
                with Timer.get_handle("summarization"):
                    self.summarize()

                # Step 3: evaluate.
                with Timer.get_handle("evaluation"):
                    self.evaluate()

                # Step 4: tick particles using timestep
                with Timer.get_handle("timestep"):
                    self.timestep()

                if check_accuracy:
                    self.check_accuracy()

        Timer.print()

    def backup_particles(self):
        self.backedup_particles = self.particles.copy()

    def check_accuracy(self):
        """Sample SAMPLE_SIZE points from the pre-computation backed up points,
        run the n^2 on them and check the difference between it and the actual
        algorithm ran
        """
        import statistics
        sample = [random.choice(range(self.n_particles)) for _ in range(SAMPLE_SIZE)]

        G = float(Config.get("bh", "grav_constant"))
        uns = rfn.structured_to_unstructured

        for j in range(self.n_particles):
            self.backedup_particles[j]['ax'] = 0
            self.backedup_particles[j]['ay'] = 0

        for i in sample:
            for j in range(self.n_particles):
                # skip if same particle
                if i == j:
                    continue

                p1_p    = uns(self.backedup_particles[i][['px', 'py']])
                p1_mass = uns(self.backedup_particles[i][['mass']])

                p2_p    = uns(self.backedup_particles[j][['px', 'py']])
                p2_mass = uns(self.backedup_particles[j][['mass']])

                dif = p1_p - p2_p
                dist = np.sqrt(np.sum(np.square(dif)))
                f = (G * p1_mass * p2_mass) / (dist*dist)

                self.backedup_particles[i]['ax'] -= f * dif[0] / p1_mass
                self.backedup_particles[i]['ay'] -= f * dif[1] / p1_mass

        cum_err = np.zeros(2)
        for i in sample:
            a1 = uns(self.backedup_particles[i][['ax', 'ay']])
            a2 = uns(self.particles[i][['ax', 'ay']])
            diff = np.abs(a1 - a2)
            print(f"diff on point {i}:  {diff}")

            cum_err += diff

        cum_err /= float(SAMPLE_SIZE)
        print(f"avg error across {SAMPLE_SIZE} points: {cum_err}")