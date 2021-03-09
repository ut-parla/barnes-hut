from barneshut.implementations.quadtree import BaseNode
from .base import BaseBarnesHut
from barneshut.internals.config import Config
from timer import Timer
from parla import Parla
from parla.cpu import *
from parla.tasks import *
import numpy as np
from itertools import combinations

class ParlaBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = BaseNode(self.size, 0, 0)
            for particle in self.particles:
                self.root_node.add_particle(particle)

    def run(self, n_iterations, print_particles=False):
        with Parla():
            partitions = Config.getint("parla", "partitions", fallback=1)
            print(f"Using {partitions} partitions")
            @spawn()
            async def main():
                # time whole run            
                with Timer.get_handle("whole_run"):
                    for _ in range(n_iterations):
                        B = TaskSpace("B")
                        # (re)create the tree for the next step
                        self.create_tree()

                        leaves = []
                        self.root_node.find_leaves(leaves)
                        print(f"we have {len(leaves)} leaves")

                        # time each iteration
                        with Timer.get_handle("iteration"):
                            chunks = np.array_split(list(combinations(leaves, 2)), partitions)
                            for i, pairs in enumerate(chunks):
                                @spawn(B[i])
                                def particle_force():
                                    for l1,l2 in pairs:
                                        if l1 is not l2:
                                            l1.apply_force(l2)
                            # Wait for them all
                            await B

                Timer.reset_and_print()
