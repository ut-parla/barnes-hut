import numpy as np
from numpy.lib import recfunctions as rfn
import random
from timer import Timer
from barneshut.internals.particle import Particle, particle_type
from barneshut.internals import Config

SAMPLE_SIZE = 10

class BaseBarnesHut:

    def __init__(self):
        self.particles = None
        self.n_particles = None
        self.backedup_particles = None

    def read_particles_from_file(self, filename):
        """Read particle coordinates, mass and initial velocity
        from file. The order of things depends on how it was generated,
        see bin/gen_input.py for details.
        """
        with open(filename) as fp:
            self.n_particles = int(fp.readline())
            self.particles = np.empty((self.n_particles,), dtype=particle_type)

            # read all lines, one particle per line
            for i in range(self.n_particles):
                self.particles[i] = Particle.particle_from_line(fp.readline())

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

    def run(self, n_iterations, partitions=None, print_particles=False, check_accuracy=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""

        with Timer.get_handle("end-to-end"):
            for _ in range(n_iterations):
                # If we're checking accuracy, we need to save points before calculation
                if check_accuracy:
                    self.backup_particles()

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

                if check_accuracy:
                    self.check_accuracy()

        Timer.print()

    def print_particles(self):
        """Print all particles' coordinates for debugging"""
        #for p in self.particles:
        #    print(repr(p))
        # TODO
        raise NotImplementedError()

    def backup_particles(self):
        self.backedup_particles = self.particles.copy()

    def check_accuracy(self):
        """Sample SAMPLE_SIZE points from the pre-computation backed up points,
        run the n^2 on them and check the difference between it and the actual
        algorithm ran
        """
        import statistics
        sample = [random.choice(range(self.n_particles)) for _ in range(SAMPLE_SIZE)]

        G = float(Config.get("bh", "grav_constant"))
        uns = rfn.structured_to_unstructured

        for j in range(self.n_particles):
            self.backedup_particles[j]['ax'] = 0
            self.backedup_particles[j]['ay'] = 0

        for i in sample:
            for j in range(self.n_particles):
                # skip if same particle
                if i == j:
                    continue
                
                p1_p    = uns(self.backedup_particles[i][['px', 'py']])
                p1_mass = uns(self.backedup_particles[i][['mass']])

                p2_p    = uns(self.backedup_particles[j][['px', 'py']])
                p2_mass = uns(self.backedup_particles[j][['mass']])

                dif = p1_p - p2_p
                dist = np.sqrt(np.sum(np.square(dif)))
                f = (G * p1_mass * p2_mass) / (dist*dist)

                self.backedup_particles[i]['ax'] -= f * dif[0] / p1_mass
                self.backedup_particles[i]['ay'] -= f * dif[1] / p1_mass

        cum_err = np.zeros(2)
        for i in sample:
            a1 = uns(self.backedup_particles[i][['ax', 'ay']])
            a2 = uns(self.particles[i][['ax', 'ay']])
            diff = np.abs(a1 - a2)
            print(f"diff on point {i}:  {diff}")
            
            cum_err += diff

        cum_err /= float(SAMPLE_SIZE)
        print(f"avg error across {SAMPLE_SIZE} points: {cum_err}")
