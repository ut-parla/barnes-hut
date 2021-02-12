from barneshut.implementations.quadtree.base import BaseNode
from .base import BaseBarnesHut
from timer import Timer
import numpy as np

class SimpleBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = BaseNode(self.size, 0, 0)
            for particle in self.particles:
                # TODO: parallelize ticks, this will be FAST
                particle.tick()
                self.root_node.add_particle(particle)

    def run(self, n_iterations, partitions=None, print_particles=False):
        # time whole run
        with Timer.get_handle("whole_run"):
            for _ in range(n_iterations):
                # (re)create the tree for the next step
                self.create_tree()

                # time each iteration
                with Timer.get_handle("iteration"):
                    # calc changes due to gravity
                    map(self.root_node.apply_gravity, self.particles)

        Timer.reset_and_print()
        if print_particles:
            self.print_particles()