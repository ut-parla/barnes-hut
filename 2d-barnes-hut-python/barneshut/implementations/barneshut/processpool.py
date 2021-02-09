from barneshut.implementations.quadtree import ProcessPoolNode
from .base import BaseBarnesHut
from timer import Timer
from concurrent.futures import ProcessPoolExecutor as MPool
#from concurrent.futures import ThreadPoolExecutor as MPool

class ProcessPoolBarnesHut (BaseBarnesHut):

    def __init__(self, n_workers=4):
        super().__init__()
        self.n_workers = n_workers

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = ProcessPoolNode(self.size, 0, 0, self.executor)
            for particle in self.particles:
                # while we're doing this, update particle positions
                # based on forces received
                particle.tick()
                self.root_node.add_particle(particle)

    def run(self, n_iterations):
        with MPool(max_workers=self.n_workers) as executor:
            self.executor = executor
            # time whole run
            with Timer.get_handle("whole_run"):
                for _ in range(n_iterations):
                    # (re)create the tree for the next step
                    self.create_tree()

                    # time each iteration
                    with Timer.get_handle("iteration"):
                        # calc changes due to gravity
                        self.root_node.recurse_to_nodes(self.particles)

            #self.executor = None

        Timer.reset_and_print()
        self.print_particles()
        #self.executor.shutdown(True, cancel_futures=True)