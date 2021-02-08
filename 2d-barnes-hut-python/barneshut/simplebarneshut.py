from .node import Node
from .particle import Particle
from timer import Timer

PRINT_POSITIONS = False

class SimpleBarnesHut:

    def __init__(self):
        self.particles = []

    def read_particles_from_file(self, filename):
        with open(filename) as fp:
            space_side_len = int(fp.readline())
            self.size = (space_side_len, space_side_len)
            self.n_particles = int(fp.readline())
            # read all lines, one particle per line
            for _ in range(self.n_particles):
                p = Particle.from_line(fp.readline())
                self.particles.append(p)
            # now that we have the particles, build tree
            self.create_tree()

    def create_tree(self):
        with Timer.get_handle("create_tree"):
            self.root_node = Node(self.size, 0, 0)
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

        # TODO: output positions
