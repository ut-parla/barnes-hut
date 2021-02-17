from barneshut.internals import Particle, Cloud
from barneshut.internals.config import Config
from collections.abc import Iterable
import numpy as np

class BaseNode:

    def __init__(self, size, x, y):
        self.width = size[0]
        self.height = size[1]
        self.x = x
        self.y = y
        self.child_nodes = {"ne": None, "se": None, "sw": None, "nw": None}
        self.cloud = Cloud()
        # a COM is now a Particle
        self.theta = Config.get("bh", "theta")

    # utility method for classes that inherit us to create their own type children
    def create_new_node(self, *args):
        return BaseNode(*args)

    def tick(self):
        self.cloud.tick_particles()

    def find_leaves(self, leaves):
        if self.is_leaf():
            leaves.append(self)
        else:
            for node in self.child_nodes.values():
                node.find_leaves(leaves)

    def add_particle(self, new_particle):
        # if we are leaf and have space, add to our set
        if self.is_leaf() and not self.cloud.is_full():
            self.cloud.add_particle(new_particle)

        # otherwise add to children:   (leaf, full) or (not leaf, not full)
        else:
            # if we don't have children node setup, create them
            if self.is_leaf():
                self.create_children()

            #if we got here we need to flush all particles to children
            if not self.cloud.is_empty():
                self.add_particle_to_children(self.cloud.particles)
                self.cloud = Cloud()

            #now add the new one
            self.add_particle_to_children(np.array([new_particle]))

    def add_particle_to_children(self, particles):
        for p in particles:
            for node in self.child_nodes.values():
                if node.bounds_around(p):
                    node.add_particle(p)
                    break
            else:
                # Node has fallen out of bounds, so we just eat it
                print ('Node moved out of bounds')

    # initially other_node is the root
    def apply_gravity(self, other_node):
        if self.cloud.is_empty():
            return

        # if empty, just return
        if other_node.is_leaf():
            if other_node.cloud.is_empty():
                return
            else:
                use_COM = self.approximation_distance(other_node)
                self.cloud.apply_force(other_node.cloud, use_COM)
        #we dont need to recurse since we are interacting leaf to leaf already

    # this checks if nodes are neighbors
    def approximation_distance(self, other_node):
        corners1 = self.get_corners()
        corners2 = other_node.get_corners()

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

    def get_corners(self):
        x1, x2 = self.x, self.x + self.width
        y1, y2 = self.y, self.y + self.height
        return ((x1, x2), (y1, y2))

    def create_children(self):
        subW = self.width / 2
        subH = self.height / 2
        subSize = (subW, subH)
        x = self.x
        y = self.y
        self.child_nodes["nw"] = self.create_new_node(subSize, x, y)
        self.child_nodes["ne"] = self.create_new_node(subSize, x + subW, y)
        self.child_nodes["se"] = self.create_new_node(subSize, x + subW, y + subH)
        self.child_nodes["sw"] = self.create_new_node(subSize, x, y + subH)

    def bounds_around(self, particle):
        x, y = particle[0:1], particle[1:2]
        return (x >= self.x
                and y >= self.y
                and x < self.x + self.width
                and y < self.y + self.height)

    def is_leaf(self):
        return self.child_nodes['ne'] is None

    def __repr__(self):
        return '<Node x: {}, y:{}, width:{}, height:{}, particle:{}, nodes:{}>'.format(self.x, self.y, self.width, self.height, self.particle, self.nodes)
