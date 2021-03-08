from barneshut.internals.particle import Particle
from timer import Timer

class BaseBarnesHut:

    def __init__(self):
        self.particles = []

    def read_particles_from_file(self, filename):
        """Read particle coordinates, mass and initial velocity
        from file. The order of things depends on how it was generated,
        see bin/gen_input.py for details.
        """
        with open(filename) as fp:
            space_side_len = int(fp.readline())
            self.size = (space_side_len, space_side_len)
            self.n_particles = int(fp.readline())
            # read all lines, one particle per line
            for _ in range(self.n_particles):
                p = Particle.particle_from_line(fp.readline())
                self.particles.append(p)

    def create_tree(self):
        """Each implementation must have it's own create_tree"""
        raise NotImplementedError()

    def summarize(self):
        """Each implementation must have it's own summarize"""
        raise NotImplementedError()

    def evaluate(self):
        """Each implementation must have it's own evaluate"""
        raise NotImplementedError()

    def timestep(self):
        """Each implementation must have it's own timestep"""
        raise NotImplementedError()

    def run(self, n_iterations, partitions=None, print_particles=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""
        with Timer.get_handle("end-to-end"):
            for _ in range(n_iterations):
                # Step 1: create tree (if Barnes-Hut), group-by points by box (if Decomposition)
                with Timer.get_handle("tree-creation"):
                    self.create_tree()

                # Step 2: summarize.
                with Timer.get_handle("summarization"):
                    self.summarize()

                # Step 3: evaluate.
                with Timer.get_handle("evaluation"):
                    self.evaluate()

                # Step 4: tick particles using timestep
                with Timer.get_handle("timestep"):
                    self.timestep()

    def print_particles(self):
        """Print all particles' coordinates for debugging"""
        for p in self.particles:
            print(repr(p))