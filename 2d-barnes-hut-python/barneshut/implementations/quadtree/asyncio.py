from . import BaseNode
import asyncio
from collections.abc import Iterable

class AsyncNode(BaseNode):

    async def apply_gravity_to_partition(self, arg):
        # if it's not a list, create one
        if not isinstance(arg, Iterable):
            particles = [arg,]
        else:
            particles = arg

        for p in particles:
            self.apply_gravity(p)

        #tasks = [self.apply_gravity(p) for p in particles]
        #await asyncio.gather(*tasks)


    #whoever is calling this is passing root as self
    def apply_gravity(self, particle):
        #if both particles are the same or there is no particle in self
        if (self.particle is particle or self.is_empty()):
            return
        #if self is this is a leaf node with particle
        elif (self.is_external_node()):
            Force.applyForceBy(particle, self.particle)
        #if particle is far enough that we can approximate
        elif (self.approximation_distance(particle)):
            Force.applyForceByCOM(particle, self.centre_of_mass)
        #if self is internal, aka has children, recurse
        else:
            # Recurse through child nodes to get more precise total force
            # Create all coroutines into an array
            #tasks = [p.apply_gravity(particle) for p in self.child_nodes.values()]
            #await asyncio.gather(*tasks)

            for child in self.child_nodes.values():
                child.apply_gravity(particle)

