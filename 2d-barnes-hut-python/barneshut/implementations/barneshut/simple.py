from barneshut.implementations.quadtree.base import BaseNode
from .base import BaseBarnesHut
from timer import Timer

PRINT_POSITIONS = False


class SimpleBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = BaseNode(self.size, 0, 0)
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

                if PRINT_POSITIONS:
                    print(f"Iteration {i}")
                    for p in self.particles:
                        print(repr(p))
                    print("\n")

        Timer.reset_and_print()