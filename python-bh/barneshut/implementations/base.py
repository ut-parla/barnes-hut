import numpy as np
import struct
from numpy.lib.recfunctions import structured_to_unstructured as unst
from numba import njit
import random
import logging
from timer import Timer
import barneshut.internals.particle as p
from barneshut.grid_decomposition import Box
from barneshut.internals import Config
from barneshut.kernels.helpers import get_bounding_box, next_perfect_square

LEAF_OCCUPANCY = 0.7

#@njit
def nsquared_acc_check(particles, samples, G):
    for i in range(samples.shape[0]):
        for j in range(particles.shape[0]):
            # skip self to self
            sample_id = samples[i, p.pid]
            if sample_id == j:
                continue
            p1_p    = samples[i, p.px:p.py+1]
            p1_mass = samples[i, p.mass]
            p2_p    = particles[j, p.px:p.py+1]
            p2_mass = particles[j, p.mass]
            dif = p1_p - p2_p
            dist = np.sqrt(np.sum(np.square(dif)))
            f = (G * p1_mass * p2_mass) / (dist*dist)

            samples[i, p.ax] -= f * dif[0] / p1_mass
            samples[i, p.ay] -= f * dif[1] / p1_mass

class BaseBarnesHut:

    def __init__(self):
        self.particles = None
        self.n_particles = None
        # Implementaitons might check this flag to do specific things
        self.checking_accuracy = False
        self.sample_size = int(Config.get("general", "sample_check_size"))
        self.particles_per_leaf = int(Config.get("grid", "max_particles_per_box"))

        self.skip_timestep = bool(Config.get("general", "skip_timestep")) 
        self.evaluation_rounds = int(Config.get("general", "evaluation_rounds")) 

        logging.debug(f"Skiping timestep? {self.skip_timestep}")

    def read_particles_from_file(self, filename):
        """Read particle coordinates, mass and initial velocity
        from file. The order of things depends on how it was generated,
        see bin/gen_input.py for details.
        """
        nsize = struct.calcsize("i")
        p_fmt = "ddddd"
        p_len = struct.calcsize(p_fmt)
        p_unpack = struct.Struct(p_fmt).unpack_from

        with open(filename, "rb") as fp:
            nb = fp.read(nsize)
            self.n_particles = struct.unpack("i", nb)[0]
            self.particles = np.empty((self.n_particles,p.nfields), dtype=np.float64)

            for i in range(self.n_particles):
                data = fp.read(p_len)
                pt = p_unpack(data)
                self.particles[i] = pt + ((0,) * (p.nfields-6)) + (float(i),)

            # particle_id = .0
            # left = self.n_particles
            # while left != 0:
            #     read_size = min(left, CHUNK_READ)
            #     left -= read_size

            #     data = fp.read(read_size*p_len)
            #     for i in range(read_size):
            #         pt = p_unpack(data[i*p_len : (i+1)*p_len])
            #         self.particles[i] = pt + ((0,) * (p.nfields-6)) + (particle_id,)
            #         particle_id += 1
        
        #import sys
        #sys.exit(0)

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

    def run(self, check_accuracy=False):
        """Runs the n-body algorithm using basic mechanisms. If
        something more intricate is required, then this method should be
        overloaded."""
        n_iterations = int(Config.get("general", "rounds"))
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
                with Timer.get_handle("grid_creation"):
                    self.create_tree()

                # Step 2: summarize.
                with Timer.get_handle("summarization"):
                    self.summarize()

                # Step 3: evaluate.
                for _ in range(self.evaluation_rounds):
                    with Timer.get_handle("evaluation"):
                        self.evaluate()

                if not self.skip_timestep:
                    # Step 4: tick particles using timestep
                    with Timer.get_handle("timestep"):
                        self.timestep()

                # At this point the implementation executed the algorithm, so we
                # get the results and compare to ours
                if self.checking_accuracy:
                    self.ensure_particles_id_ordered()
                    self.check_accuracy(sample_indices, nsquared_sample)

        self.cleanup()

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
        samples_nd = np.zeros((len(sample_indices), p.nfields))
        # copy sampled particles so we can modify them
        for i, si in enumerate(sample_indices):
            samples_nd[i] = particles[si].copy()
            samples_nd[i, p.ax] = 0
            samples_nd[i, p.ay] = 0
        
        G = float(Config.get("bh", "grav_constant"))
        # for each particle, do the n^2 algorithm
        nsquared_acc_check(particles, samples_nd, G)
        logging.debug(f"n^2 algo for particle {i}: {samples_nd[:,p.ax:p.ay+1]}")

        samples = {}
        for i in range(samples_nd.shape[0]):
            pid = samples_nd[i, p.pid]
            samples[pid] = samples_nd[i].copy()

        return samples

    def check_accuracy(self, sample_indices, nsquared_sample):
        """Sample SAMPLE_SIZE points from the pre-computation backed up points,
        run the n^2 on them and check the difference between it and the actual
        algorithm ran
        """
        impl_sample = self.get_particles(sample_indices)
        cum_err = np.zeros(2)
        for i in sample_indices:
            nsq = nsquared_sample[i][p.ax:p.ay+1]
            aprox = impl_sample[i][p.ax:p.ay+1]
            
            id1 = nsquared_sample[i][p.pid]
            id2 = impl_sample[i][p.pid]
            assert id1 == id2

            rel_err = np.fabs((nsq - aprox) / nsq)

            logging.debug(f"nsquared p{i} : {nsq}")
            logging.debug(f"impl     p{i} : {aprox}")
            logging.debug(f"relative err:  {rel_err}")
            cum_err += rel_err

        cum_err /= float(self.sample_size)
        err = ", ".join([str(x) for x in cum_err])
        print(f"avg relative {self.sample_size} points on each axis:\n\t{err}")

    #
    # Some common helper methods
    #

    def set_particles_bounding_box(self):
        # get square bounding box around all particles
        #unstr_points = unst(self.particles[['px', 'py']], copy=False)
        
        bb_min, bb_max = get_bounding_box(self.particles[:,p.px:p.py+1])
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
        logging.warning(f'''With {LEAF_OCCUPANCY} occupancy, {self.particles_per_leaf} particles per leaf 
                we need {nleaves} leaves, whose next perfect square is {self.grid_dim}.
                Grid will be {self.grid_dim}x{self.grid_dim}''')
    
        # set BB points and step
        self.step = (top_right[0] - bottom_left[0]) / self.grid_dim 
        self.min_xy = bottom_left
        self.max_xy = top_right

    def create_grid_boxes(self):
        """Use bounding boxes coordinates, create the grid
        matrix and their boxes
        """
        
        # x and y have the same edge length, so get x length
        step = (self.max_xy[0]-self.min_xy[0]) / self.grid_dim
        # create grid as a matrix, starting from bottom left
        self.grid = []
        logging.debug(f"Grid: {self.min_xy}, {self.max_xy}")
        for i in range(self.grid_dim):
            row = []
            for j in range(self.grid_dim):
                x = self.min_xy[0] + (i*step)
                y = self.min_xy[1] + (j*step)
                row.append(Box((x,y), (x+step, y+step)))
                #logging.debug(f"Box {i}/{j}: {(x,y)}, {(x+step, y+step)}")
            self.grid.append(row)