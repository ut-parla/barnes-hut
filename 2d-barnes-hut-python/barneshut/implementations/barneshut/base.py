from barneshut.internals.particle import Particle


class BaseBarnesHut:

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
        raise NotImplementedError()

    def run(self, n_iterations):
        raise NotImplementedError()
