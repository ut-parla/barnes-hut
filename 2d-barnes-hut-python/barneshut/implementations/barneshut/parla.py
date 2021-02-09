from barneshut.implementations.quadtree import BaseNode
from .base import BaseBarnesHut
from timer import Timer
from parla import Parla
from parla.cpu import *
from parla.tasks import *

class ParlaBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = BaseNode(self.size, 0, 0)
            for particle in self.particles:
                particle.tick()
                self.root_node.add_particle(particle)

    def __run(self, n_iterations):
        @spawn()
        async def main():
            # time whole run
            
            with Timer.get_handle("whole_run"):
                for _ in range(n_iterations):
                    B = TaskSpace("B")
                    # (re)create the tree for the next step
                    self.create_tree()
                    # time each iteration
                    with Timer.get_handle("iteration"):
                        for i, p in enumerate(self.particles):
                            @spawn(B[i])
                            def particle_force():
                                self.root_node.applyGravityTo(p)
                        # Wait for them all
                        await B
                        #@spawn(dependencies=B)
                        #def wait():
                        #    pass
                        #await wait

            Timer.reset_and_print()
            self.print_particles()
            
    def run(self, n_iterations):
        with Parla():
            self.__run(n_iterations)