import numpy as np
from numpy.lib import recfunctions as rfn
import logging
from .base import BaseBarnesHut
from barneshut.internals.config import Config
from barneshut.grid_decomposition import Box
from barneshut.kernels.helpers import next_perfect_square, get_bounding_box
from timer import Timer
from numpy import sqrt

LEAF_OCCUPANCY = 0.7

class SequentialBarnesHut (BaseBarnesHut):
    """ Sequential implementation of nbody. Currently not Barnes-hut but
    a box decomposition."""

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
        self.grid = None

        # setup functions
        from barneshut.kernels.grid_decomposition.sequential.grid import get_grid_placement_fn
        self.__grid_placement = get_grid_placement_fn()

        from barneshut.kernels.grid_decomposition.sequential.evaluation import get_evaluation_fn
        self.__evaluate = get_evaluation_fn()
        
    def __create_grid(self, bottom_left, top_right, grid_dim):
        """Use bounding boxes coordinates, create the grid
        matrix and their boxes
        """
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
                row.append(Box((x,y), (x+step, y+step)))
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
            placements = self.__grid_placement(points, bottom_left, step, grid_dim)

        with Timer.get_handle("grid assign"):
            for i in range(len(placements)):
                x, y = placements[i].astype(int, copy=False)
                logging.debug(f"adding point {i} ({self.particles[i]}) to box {x}/{y}")
                self.grid[x][y].add_particle(self.particles[i])

    def summarize(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].get_COM()

    def evaluate(self):
        # TODO: do i,i evaluations in all kernels
        # TODO: use the update flag on the kernels to save a lot of computation
        self.__evaluate(self.grid)

    def timestep(self):
        n = len(self.grid)
        for i in range(n):
            for j in range(n):
                self.grid[i][j].tick()

    # def __get_bounding_box_manual(self):
    #     # find bounding box; min/max coordinates on each axis
    #     max_x, min_x = -1, -1
    #     max_y, min_y = -1, -1
    #     for p in self.particles:
    #         # please dont hate, i just wanna save some lines
    #         max_x = p['px'] if p['px'] > max_x or max_x == -1 else max_x
    #         min_x = p['px'] if p['px'] < min_x or min_x == -1 else min_x
    #         max_y = p['py'] if p['py'] > max_y or max_y == -1 else max_y
    #         min_y = p['py'] if p['py'] < min_y or min_y == -1 else min_y
    #     assert max_x != -1 and min_x != -1 and max_y != -1 and min_y != -1

    #     # find longer edge and increase the shorter so we have a square
    #     x_edge, y_edge = max_x - min_x, max_y - min_y 
    #     if x_edge >= y_edge:
    #         max_y += (x_edge - y_edge)
    #     else:
    #         max_x += (y_edge - x_edge)

    #     # assert it is a square
    #     assert (max_x-min_x)==(max_y-min_y)
    #     return (min_x, min_y), (max_x, max_y)

    # # TODO: we might need a testing framework.
    # # for now, np is much faster:
    # # name, avg, stddev
    # # manual, 4.293787787999463, 0.0
    # # np, 0.14757852000002458, 0.0
    # def test_bb(self):
    #     with Timer.get_handle("manual"):
    #         for i in range(1000):
    #             self.__get_bounding_box_manual()
    #     with Timer.get_handle("np"):
    #         for i in range(1000):
    #             self.__get_bounding_box()
    #     Timer.print()
    #     m = self.__get_bounding_box_manual()
    #     n = self.__get_bounding_box()
    #     print(m)
    #     print(n)

    # def run(self, n_iterations, partitions=None, print_particles=False):
    #     self.test_bb()