from barneshut.internals import constants
from barneshut.internals.centreofmass import CentreOfMass
from . import BaseNode


class ProcessPoolNode (BaseNode):

    def __init__(self, size, x, y, executor):
        super().__init__(size, x, y)
        self.executor = executor

    def recurse_to_nodes(self, particles):
        self.executor.map(self.applyGravityTo, particles)
        #map(self.applyGravityTo, particles)
    
    def populate_nodes(self):
        subW = self.width / 2
        subH = self.height / 2
        subSize = (subW, subH)
        x = self.x
        y = self.y
        self.child_nodes["nw"] = ProcessPoolNode(subSize, x, y, self.executor)
        self.child_nodes["ne"] = ProcessPoolNode(subSize, x + subW, y, self.executor)
        self.child_nodes["se"] = ProcessPoolNode(subSize, x + subW, y + subH, self.executor)
        self.child_nodes["sw"] = ProcessPoolNode(subSize, x, y + subH, self.executor)

    #whoever is calling this is passing root as self
    def applyGravityTo(self, particle):
        #if both particles are the same or there is no particle in self
        if (self.particle is particle or self.isEmptyNode()):
            return
        #if self is this is a leaf node with particle
        elif (self.isExternalNode()):
            Force.applyForceBy(particle, self.particle)
        #if particle is far enough that we can approximate
        elif (self.isFarEnoughForApproxMass(particle)):
            Force.applyForceByCOM(particle, self.centre_of_mass)
        #if self is internal, aka has children, recurse
        else:
            # Recurse through child nodes to get more precise total force
            # futures = []
            # for child in self.child_nodes.values():
            #     fut = self.executor.submit(child.applyGravityTo, particle)
            #     futures.append(fut)
            # #wait for all
            # for fut in futures:
            #     fut.result()

            for child in self.child_nodes.values():
                child.applyGravityTo(particle)
                