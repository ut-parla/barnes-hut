from . import BaseNode


class ProcessPoolNode (BaseNode):

    def __init__(self, size, x, y, executor):
        super().__init__(size, x, y)
        self.executor = executor

    def recurse_to_nodes(self, particles):
        self.executor.map(self.apply_gravity, particles)
        #map(self.apply_gravity, particles)
    
    def create_children(self):
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
            # futures = []
            # for child in self.child_nodes.values():
            #     fut = self.executor.submit(child.apply_gravity, particle)
            #     futures.append(fut)
            # #wait for all
            # for fut in futures:
            #     fut.result()

            for child in self.child_nodes.values():
                child.apply_gravity(particle)
                