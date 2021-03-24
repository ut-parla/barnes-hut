import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as unst
import random
import logging
from timer import Timer
from barneshut.internals.particle import Particle, particle_type
from barneshut.internals import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square

LEAF_OCCUPANCY = 0.7

class BaseBarnesHut:

    def __init__(self):
        self.particles = None
        self.n_particles = None
        # Implementaitons might check this flag to do specific things
        self.checking_accuracy = False
        self.sample_size = int(Config.get("general", "sample_check_size"))
        self.particles_per_leaf = int(Config.get("quadtree", "particles_per_leaf"))

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

    def cleanup(self):
        """In case we need to do cleanup after we run, override this."""
        pass

    def run(self, n_iterations, partitions=None, print_particles=False, check_accuracy=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""
        self.checking_accuracy = check_accuracy

        if self.checking_accuracy:
            sample_indices = self.generate_sample_indices()
        
        with Timer.get_handle("end-to-end"):
            for _ in range(n_iterations):
                # If we're checking accuracy, we need to tell the implementation
                # and also get the particles before they are modified
                if self.checking_accuracy:
                    nsquared_sample = self.preround_accuracy_check(sample_indices)

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

                # At this point the implementation executed the algorithm, so we
                # get the results and compare to ours
                if self.checking_accuracy:
                    self.check_accuracy(sample_indices, nsquared_sample)

        Timer.print()
        self.cleanup()

    def print_particles(self):
        """Print all particles' coordinates for debugging"""
        #for p in self.particles:
        #    print(repr(p))
        # TODO
        raise NotImplementedError()

    def get_particles(self, sample_indices=None):
        """ Implementations that want correctness check MUST override
        this method. Must return the list of particles in the same
        original order it started.
        If sample_indices is None, return all particles.
        Else return a dict with {sample_index: particle, ...}.
        For example, the sequential implementation modifies the
        array in place, so it must unsort the array before returning.
        Doesn't need to be a copy, can be a slice/reference.
        """
        raise NotImplementedError()

    def generate_sample_indices(self):
        # fixed seed just so we keep this equal across runs
        random.seed(0)
        samples = random.sample(range(self.n_particles), self.sample_size)
        logging.debug(f"acc_check: sample is {samples}")
        return samples

    def preround_accuracy_check(self, sample_indices):
        """ Get the original particles, run the n^2 algorithm
        for a sample and store the results.
        After the implementation runs, we compare.
        """
        particles = self.get_particles()
        samples = {}

        # copy sampled particles so we can modify them
        for i in sample_indices:
            samples[i] = particles[i].copy()
            samples[i]['ax'] = 0
            samples[i]['ay'] = 0

        G = float(Config.get("bh", "grav_constant"))
        # for each particle, do the n^2 algorithm
        for i in sample_indices:
            for j in range(len(particles)):
                # skip self to self
                if i == j:
                    continue
                p1_p    = unst(samples[i][['px', 'py']])
                p1_mass = unst(samples[i][['mass']])
                p2_p    = unst(particles[j][['px', 'py']])
                p2_mass = unst(particles[j][['mass']])
                dif = p1_p - p2_p
                dist = np.sqrt(np.sum(np.square(dif)))
                f = (G * p1_mass * p2_mass) / (dist*dist)

                samples[i]['ax'] -= f * dif[0] / p1_mass
                samples[i]['ay'] -= f * dif[1] / p1_mass

            logging.debug(f"n^2 algo for particle {i}: {samples[i]['ax']}/{samples[i]['ay']}")

        tick = float(Config.get("bh", "tick_seconds"))
        for i in sample_indices:

            samples[i]['vx'] += samples[i]['ax'] * tick
            samples[i]['vy'] += samples[i]['ay'] * tick
            samples[i]['ax'] = 0.0
            samples[i]['ay'] = 0.0
            samples[i]['px'] += samples[i]['vx'] * tick
            samples[i]['py'] += samples[i]['vy'] * tick
        return samples

    def check_accuracy(self, sample_indices, nsquared_sample):
        """Sample SAMPLE_SIZE points from the pre-computation backed up points,
        run the n^2 on them and check the difference between it and the actual
        algorithm ran
        """
        impl_sample = self.get_particles(sample_indices)
        cum_err = np.zeros(2)
        for i in sample_indices:
            nsq = unst(nsquared_sample[i][['px', 'py']])
            a2 = unst(impl_sample[i][['px', 'py']])
            
            diff = np.fabs(nsq - a2) / nsq

            logging.debug(f"nsquared p{i} : {nsq}")
            logging.debug(f"impl     p{i} : {a2}")
            logging.debug(f"diff on point {i}:  {np.fabs(nsq - a2) / nsq}")
            logging.debug(f"relative err:  {diff}")
            cum_err += diff

        cum_err /= float(self.sample_size)
        err = ", ".join([str(x) for x in cum_err])
        print(f"avg relative error across {self.sample_size} points: {err}")


    #
    # Some common helper methods
    #

    def set_particles_bounding_box(self):
        # get square bounding box around all particles
        unstr_points = unst(self.particles[['px', 'py']], copy=False)
        bb_min, bb_max = get_bounding_box(unstr_points)
        bottom_left = np.array(bb_min)
        top_right = np.array(bb_max)

        # if more than one particle per leaf, let's assume an occupancy of
        # 80% (arbitrary number), because if we use 100% we might have leaves
        # with >particles_per_leaf particles. This is all assuming a normal
        # random distribution.
        if self.particles_per_leaf == 1:
            nleaves = self.particles_per_leaf
        else:
            n = len(self.particles)
            nleaves = n / (LEAF_OCCUPANCY * self.particles_per_leaf)

        # find next perfect square
        nleaves = next_perfect_square(nleaves)
        self.grid_dim = int(nleaves**0.5)
        logging.debug(f'''With {LEAF_OCCUPANCY} occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {self.grid_dim}.
                Grid will be {self.grid_dim}x{self.grid_dim}''')
    
        # set BB points and step
        self.step = (top_right[0] - bottom_left[0]) / self.grid_dim 
        self.min_xy = bottom_left
        self.max_xy = top_right