from barneshut.implementations.quadtree import AsyncNode
from .base import BaseBarnesHut
from timer import Timer
import asyncio

class AsyncBarnesHut (BaseBarnesHut):

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = AsyncNode(self.size, 0, 0)
            for particle in self.particles:
                # while we're doing this, update particle positions
                # based on forces received
                particle.tick()
                self.root_node.add_particle(particle)

    async def __run(self, n_iterations):
        # time whole run
        with Timer.get_handle("whole_run"):
            for _ in range(n_iterations):
                # time each iteration
                with Timer.get_handle("iteration"):
                    # calc changes due to gravity
                    tasks = [self.root_node.applyGravityTo(p) for p in self.particles]
                    # Wait for them all
                    await asyncio.gather(*tasks)
                        
                # recreate the tree for the next step
                self.create_tree()

        Timer.reset_and_print()

    def run(self, n_iterations):
        asyncio.get_event_loop().run_until_complete(self.__run(n_iterations))