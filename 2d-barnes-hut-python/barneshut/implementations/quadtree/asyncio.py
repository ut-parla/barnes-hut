from barneshut.internals import constants
from barneshut.internals.centreofmass import CentreOfMass
from barneshut.internals.force import Force
from . import BaseNode
import asyncio

class AsyncNode(BaseNode):

    def add_particle(self, newParticle):
        if (self.isEmptyNode()):
            self.particle = newParticle
            self.centreOfMass = newParticle.getCentreOfMass()
            return

        if (self.childNodes['ne'] is None):
            self.populate_nodes()

        if (self.particle is not None):
            # clear centre of mass, as we're going to update it
            # based on child nodes
            self.centreOfMass = None

        self.add_particleToChildNodes(self.particle)
        self.add_particleToChildNodes(newParticle)
        #self.centreOfMass = centreOfMass.combine(centreOfMass2)
        # self.centreOfMass = self.centreOfMass.combine(newParticle.getCentreOfMass())
        self.particle = None

    def populate_nodes(self):
        subW = self.width / 2
        subH = self.height / 2
        subSize = (subW, subH)
        x = self.x
        y = self.y
        self.childNodes["nw"] = AsyncNode(subSize, x, y)
        self.childNodes["ne"] = AsyncNode(subSize, x + subW, y)
        self.childNodes["se"] = AsyncNode(subSize, x + subW, y + subH)
        self.childNodes["sw"] = AsyncNode(subSize, x, y + subH)

    def add_particleToChildNodes(self, particle):
        if (particle is None):
            return
        # self.totalMass += particle.mass

        for node in self.childNodes.values():
            if (node.boundsAround(particle)):
                node.add_particle(particle)
                com = node.centreOfMass
                self.centreOfMass = com.combine(self.centreOfMass)
                return

        # Node has fallen out of bounds, so we just eat it
        print ('Node moved out of bounds')

    #whoever is calling this is passing root as self
    async def applyGravityTo(self, particle):
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
            # Create all coroutines into an array
            tasks = [p.applyGravityTo(particle) for p in self.childNodes.values()]
            # Wait for them all
            await asyncio.gather(*tasks)

    def isFarEnoughForApproxMass(self, particle):
        # s('regionwidth') / d('distance') < theta
        # return False
        d = self.centreOfMass.pos.dist(particle.pos)
        return self.width / d < self.theta

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
