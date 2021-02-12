from barneshut.internals import constants
from barneshut.internals.centreofmass import CentreOfMass
from . import BaseNode
import asyncio
from collections.abc import Iterable

class AsyncNode(BaseNode):

    async def chunkedApplyGravityTo(self, arg):
        # if it's not a list, create one
        if not isinstance(arg, Iterable):
            particles = [arg,]
        else:
            particles = arg

        for p in particles:
            self.applyGravityTo(p)

        #tasks = [self.applyGravityTo(p) for p in particles]
        #await asyncio.gather(*tasks)


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
            # Create all coroutines into an array
            #tasks = [p.applyGravityTo(particle) for p in self.child_nodes.values()]
            #await asyncio.gather(*tasks)

            for child in self.child_nodes.values():
                child.applyGravityTo(particle)

