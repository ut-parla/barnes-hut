from barneshut.implementations.quadtree import AsyncNode, BaseNode
from .base import BaseBarnesHut
from timer import Timer
import asyncio
import numpy as np

class AsyncBarnesHut (BaseBarnesHut):

    def create_children(self):
        subW = self.width / 2
        subH = self.height / 2
        subSize = (subW, subH)
        x = self.x
        y = self.y
        self.child_nodes["nw"] = AsyncBarnesHut(subSize, x, y)
        self.child_nodes["ne"] = AsyncBarnesHut(subSize, x + subW, y)
        self.child_nodes["se"] = AsyncBarnesHut(subSize, x + subW, y + subH)
        self.child_nodes["sw"] = AsyncBarnesHut(subSize, x, y + subH)

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = AsyncNode(self.size, 0, 0)
            for particle in self.particles:
                # while we're doing this, update particle positions
                # based on forces received
                particle.tick()
                self.root_node.add_particle(particle)

    async def __run(self, n_iterations, partitions):
        print(f"Using {partitions} partitions")
        # time whole run
        with Timer.get_handle("whole_run"):
            for _ in range(n_iterations):
                # (re)create the tree for the next step
                self.create_tree()

                # time each iteration
                with Timer.get_handle("iteration"):
                    if partitions is None:
                        tasks = [self.root_node.apply_gravity(p) for p in self.particles]
                    else:
                        chunks = np.array_split(self.particles, partitions)
                        tasks = [self.root_node.apply_gravity_to_partition(ps) for ps in chunks]
                        # Wait for them all
                        await asyncio.gather(*tasks)

        Timer.reset_and_print()

    def run(self, n_iterations, partitions=None):
        asyncio.get_event_loop().run_until_complete(self.__run(n_iterations, partitions))