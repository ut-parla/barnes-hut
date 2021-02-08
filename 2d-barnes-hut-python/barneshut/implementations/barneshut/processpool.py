from barneshut.implementations.quadtree import ProcessPoolNode
from .base import BaseBarnesHut
from timer import Timer
from concurrent.futures import ProcessPoolExecutor


class ProcessPoolBarnesHut (BaseBarnesHut):

    def __init__(self, n_workers):
        super().__init__()
        self.executor = ProcessPoolExecutor(max_workers=n_workers)

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = ProcessPoolNode(self.size, 0, 0, self.executor)
            for particle in self.particles:
                # while we're doing this, update particle positions
                # based on forces received
                particle.tick()
                self.root_node.add_particle(particle)

    def run(self, n_iterations):
        # time whole run
        with Timer.get_handle("whole_run"):
            for i in range(n_iterations):
                # time each iteration
                with Timer.get_handle("iteration"):
                    # calc changes due to gravity
                    for particle in self.particles:
                        self.root_node.applyGravityTo(particle)

                # recreate the tree for the next step
                self.create_tree()

        Timer.reset_and_print()
