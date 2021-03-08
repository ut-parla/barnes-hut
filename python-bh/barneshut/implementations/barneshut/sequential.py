import numpy as np
import logging
from math import sqrt, pow, ceil
from .base import BaseBarnesHut
from barneshut.internals.config import Config
from barneshut.grid_decomposition import Box

class SequentialBarnesHut (BaseBarnesHut):
    """ Sequential implementation of nbody. Currently not Barnes-hut but
    a box decomposition."""

    def __init__(self):
        """Our parent will init `self.particles = []` only, we need to do 
        what else we need."""
        super().__init__()
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))
        self.grid = None

    def __next_perfect_square(self, n):
        if n%n**0.5 != 0:
            return pow( ceil(sqrt(n))  , 2)
        return n

    def __get_bounding_box(self):
        """ Get bounding box coordinates around all particles.
        Returns the bottom left and top right corner coordinates, making
        sure that it is a square.
        """
        # find bounding box; min/max coordinates on each axis
        max_x, min_x = -1, -1
        max_y, min_y = -1, -1
        for p in self.particles:
            # please dont hate, i just wanna save some lines
            max_x = p.position[0] if p.position[0] > max_x or max_x == -1 else max_x
            min_x = p.position[0] if p.position[0] < min_x or min_x == -1 else min_x
            max_y = p.position[1] if p.position[1] > max_y or max_y == -1 else max_y
            min_y = p.position[1] if p.position[1] < min_y or min_y == -1 else min_y
        assert max_x != -1 and min_x != -1 and max_y != -1 and min_y != -1

        # find longer edge and increase the shorter so we have a square
        x_edge, y_edge = max_x - min_x, max_y - min_y 
        if x_edge >= y_edge:
            max_y += (x_edge - y_edge)
        else:
            max_x += (y_edge - x_edge)

        # assert it is a square
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
                row.append(Box((x,y), (x+step, y+step)))
            logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)

        

    def create_tree(self):
        """We're not creating an actual tree, just grouping particles 
        by the box in the grid they belong.
        """
        # get bounding box around all particles
        bb_min, bb_max = self.__get_bounding_box() 
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
            nleaves = n / (0.7 * self.particles_per_leaf)

        # find next perfect square
        nleaves = self.__next_perfect_square(nleaves)
        grid_dim = int(sqrt(nleaves))

        logging.debug(f'''With 0.8 occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {grid_dim}.
                Grid will be {grid_dim}x{grid_dim}''')
        
        self.__create_grid(bottom_left, top_right, grid_dim)

        # TODO: place this in a kernel
        # TODO: use numpy to do batches
        bb_x = np.array([bottom_left[0], top_right[0]])
        # bb_y = np.array([bottom_left[1], top_right[1]])
        edge_len = bb_x[1] - bb_x[0]
        step =  edge_len / grid_dim 

        # placements is an array mapping points to their position in the matrix
        # this is just so we can easily map to numpy/cuda later
        placements = np.ndarray((len(self.particles), 2))
        for i, p in enumerate(self.particles):
            x, y = p.position[0], p.position[1]
            px, py = x/step, y/step
            placements[i][0], placements[i][1] = px, py 
        
        for i, p in enumerate(placements):
            # need to get min because of float rounding
            x = min(int(p[0]), grid_dim-1)
            y = min(int(p[1]), grid_dim-1)
            logging.debug(f"adding point {i} ({self.particles[i].position}) to box {x}/{y}")
            self.grid[x][y].add_particle(self.particles[i])

        
    def summarize(self):
        """Each implementation must have it's own summarize"""
        raise NotImplementedError()

    def evaluate(self):
        """Each implementation must have it's own evaluate"""
        raise NotImplementedError()

    def timestep(self):
        """Each implementation must have it's own timestep"""
        raise NotImplementedError()
