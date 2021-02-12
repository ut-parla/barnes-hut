from barneshut.internals import constants
from barneshut.internals.centreofmass import CentreOfMass
from collections.abc import Iterable
import numpy as np

class BaseNode:

    def __init__(self, size, x, y):
        self.width = size[0]
        self.height = size[1]
        self.x = x
        self.y = y
        self.childNodes = {"ne": None, "se": None, "sw": None, "nw": None}
        self.particle = None
        self.centreOfMass = None
        self.theta = constants.THETA

    def add_particle(self, newParticle):
        if (self.isEmptyNode()):
            self.particle = newParticle
            self.centreOfMass = newParticle.getCentreOfMass()
            if self.centreOfMass is None:
                print("NO WAY ITS NONE")
            return

        if (self.childNodes['ne'] is None):
            self.populate_nodes()

        if (self.particle is not None):
            # clear centre of mass, as we're going to update it
            # based on child nodes
            self.centreOfMass = None
            #self.centreOfMass = CentreOfMass(0, Point(0,0))

        #gotta understand this
        self.add_particleToChildNodes(self.particle)
        self.add_particleToChildNodes(newParticle)
        self.particle = None

    def populate_nodes(self):
        subW = self.width / 2
        subH = self.height / 2
        subSize = (subW, subH)
        x = self.x
        y = self.y
        self.childNodes["nw"] = BaseNode(subSize, x, y)
        self.childNodes["ne"] = BaseNode(subSize, x + subW, y)
        self.childNodes["se"] = BaseNode(subSize, x + subW, y + subH)
        self.childNodes["sw"] = BaseNode(subSize, x, y + subH)

    def add_particleToChildNodes(self, particle):
        if (particle is None):
            return

        for node in self.childNodes.values():
            if (node.boundsAround(particle)):
                node.add_particle(particle)
                com = node.centreOfMass
                #self.centreOfMass = com.combine(self.centreOfMass)
                if self.centreOfMass is None:
                    self.centreOfMass = com 
                else:
                    self.centreOfMass.combine(com)
                return

        # Node has fallen out of bounds, so we just eat it
        print ('Node moved out of bounds')

    def boundsAround(self, particle):
        x, y = particle.pX, particle.pY
        return (x >= self.x
                and y >= self.y
                and x < self.x + self.width
                and y < self.y + self.height)

    def chunkedApplyGravityTo(self, arg):
        # if it's not a list, create one
        if not isinstance(arg, Iterable):
            particles = [arg,]
        else:
            particles = arg

        for p in particles:
            self.applyGravityTo(p)

    #whoever is calling this is passing root as self
    def applyGravityTo(self, particle):
        #if both particles are the same or there is no particle in self
        if (self.particle is particle or self.isEmptyNode()):
            return
        #if self is this is a leaf node with particle
        elif (self.isExternalNode()):
            particle.apply_force(self.particle)
        #if particle is far enough that we can approximate
        elif (self.isFarEnoughForApproxMass(particle)):
            particle.apply_force_COM(self.centreOfMass)
        #if self is internal, aka has children, recurse
        else:
            # Recurse through child nodes to get more precise total force
            for child in self.childNodes.values():
                child.applyGravityTo(particle)

    def isFarEnoughForApproxMass(self, particle):
        #self.centreOfMass.pos.dist(particle.pos)
        d = particle.calculate_distance(self.centreOfMass.pX, self.centreOfMass.pY)
        return np.divide(self.width, d) < self.theta

    #No particle in this node, but there are children
    def isInternalNode(self):
        return self.particle is None and self.childNodes['ne'] is not None

    #No children, but there are particles
    def isExternalNode(self):
        return self.childNodes['ne'] is None and self.particle is not None

    #No children, no particles
    def isEmptyNode(self):
        return self.childNodes['ne'] is None and self.particle is None

    def __repr__(self):
        return '<Node x: {}, y:{}, width:{}, height:{}, particle:{}, nodes:{}>'.format(self.x, self.y, self.width, self.height, self.particle, self.nodes)
