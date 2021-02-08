from barneshut.internals import constants
from barneshut.internals.centreofmass import CentreOfMass
from barneshut.internals.force import Force
from . import BaseNode


class ProcessPoolNode (BaseNode):

    def __init__(self, size, x, y, executor):
        super().__init__(size, x, y)
        self.executor = executor

    # this could be transformed into a executor map, but i dont think its worth it
    # def create_tree(self):
    #     with Timer.get_handle("create_tree"):
    #         self.root_node = BaseNode(self.size, 0, 0)
    #         for particle in self.particles:
    #             # while we're doing this, update particle positions
    #             # based on forces received
    #             particle.tick()
    #             self.root_node.add_particle(particle)

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
            Force.applyForceByCOM(particle, self.centreOfMass)
        #if self is internal, aka has children, recurse
        else:
            # Recurse through child nodes to get more precise total force
            futures = []
            for child in self.childNodes.values():
                fut = self.executor.submit(child.applyGravityTo, particle)
                futures.append(fut)

            #wait for all
            for fut in futures:
                fut.result()