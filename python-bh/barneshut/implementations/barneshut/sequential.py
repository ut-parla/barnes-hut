from barneshut.implementations.quadtree.base import BaseNode
from .base import BaseBarnesHut
from timer import Timer
from barneshut.internals.config import Config
import numpy as np
from itertools import combinations

class SequentialBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = BaseNode(self.size, 0, 0)
            for particle in self.particles:
                self.root_node.add_particle(particle)

    def run(self, n_iterations, partitions=None, print_particles=False):
        # time whole run
        with Timer.get_handle("whole_run"):
            for _ in range(n_iterations):
                # (re)create the tree for the next step
                self.create_tree()

                # TODO: just wanna make this work, we can do this step during construction later
                leaves = []
                self.root_node.find_leaves(leaves)
                print(f"we have {len(leaves)} leaves")

                # time each iteration
                with Timer.get_handle("iteration"):
                    for l1,l2 in combinations(leaves, 2):
                        if l1 is not l2:
                            l1.apply_force(l2)
                    
                    for leaf in leaves:
                        leaf.tick()

        Timer.reset_and_print()
        if print_particles:
            self.print_particles()
