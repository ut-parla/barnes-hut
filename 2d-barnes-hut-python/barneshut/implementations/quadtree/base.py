from barneshut.internals import constants
from barneshut.internals.particle import Particle, new_zero_particle


from collections.abc import Iterable
import numpy as np


class BaseNode:

    def __init__(self, size, x, y):
        self.width = size[0]
        self.height = size[1]
        self.x = x
        self.y = y
        self.child_nodes = {"ne": None, "se": None, "sw": None, "nw": None}
        # TODO: keep multiple particles
        self.particle = None
        # a COM is now a Particle, so we can reuse the class, since it's *mostly* the same thing
        self.centre_of_mass = new_zero_particle()
        self.theta = constants.THETA

    # utility method for classes that inherit us to create their own type children
    def create_new_node(self, *args):
        return BaseNode(*args)

    def add_particle(self, new_particle):
        # empty node means we have no particles
        if (self.is_empty()):
            # TODO: add to particles
            self.particle       = new_particle
            # TODO: combine COMs
            self.centre_of_mass = new_particle
        else:
            # if we don't have children node setup, create them
            if (self.child_nodes['ne'] is None):
                self.create_children()

            # if we are leaf, clear COM, as we're going to update it based on child nodes
            if (self.particle is not None):
                self.centre_of_mass = new_zero_particle()

            # if we got here, this is a leaf node
            # so we must add our particle to our children, remove it from our particles
            # and also add the new particle
            self.add_particle_to_children(self.particle)
            self.add_particle_to_children(new_particle)
            self.particle = None

    def add_particle_to_children(self, particle):
        if particle is None:
            return

        for node in self.child_nodes.values():
            if node.bounds_around(particle):
                #add particle to the node
                node.add_particle(particle)
                #update our COM
                self.centre_of_mass.combine_COM(node.centre_of_mass)
                return

        # Node has fallen out of bounds, so we just eat it
        print ('Node moved out of bounds')

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
        x, y = particle.position[0], particle.position[1]
        return (x >= self.x
                and y >= self.y
                and x < self.x + self.width
                and y < self.y + self.height)

    def apply_gravity_to_partition(self, arg):
        # if it's not a list, create one
        if not isinstance(arg, Iterable):
            particles = [arg,]
        else:
            particles = arg

        for p in particles:
            self.apply_gravity(p)

    #whoever is calling this is passing root as self
    def apply_gravity(self, particle):
        #if both particles are the same or we're an empty leaf
        if self.particle is particle or self.is_empty():
            return

        #if self is this is a leaf node with particle
        elif self.is_external_node():
            particle.apply_force(self.particle)
        
        #if particle is far enough that we can approximate
        elif self.approximation_distance(particle):
            particle.apply_force(self.centre_of_mass, isCOM=True)
        
        #if self is internal, aka has children, recurse
        else:
            # Recurse through child nodes to get more precise total force
            for child in self.child_nodes.values():
                child.apply_gravity(particle)

    def approximation_distance(self, particle):
        distance = particle.calculate_distance(self.centre_of_mass)
        return np.divide(self.width, distance) < self.theta

    #No children, but there are particles
    def is_external_node(self):
        return self.child_nodes['ne'] is None and self.particle is not None

    #No children, no particles
    def is_empty(self):
        return self.child_nodes['ne'] is None and self.particle is None

    def __repr__(self):
        return '<Node x: {}, y:{}, width:{}, height:{}, particle:{}, nodes:{}>'.format(self.x, self.y, self.width, self.height, self.particle, self.nodes)
