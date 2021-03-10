from barneshut.internals.config import Config
from .implementations.sequential import SequentialBarnesHut
from .implementations.pykokkos import PyKokkosBarnesHut

class BarnesHut:

    def __init__(self, ini_file):
        Config.read_file(ini_file)
        impl = Config.get("general", "implementation", fallback="sequential")
        print(f"Using {impl} implementation")

        bh = None
        if impl == "sequential":
            bh = SequentialBarnesHut()
        # elif impl == "process":
        #     bh = ProcessPoolBarnesHut()
        # elif impl == "async":
        #     bh = AsyncBarnesHut()
        #elif impl == "parla":
            #bh = ParlaBarnesHut()
            
        self.bh = bh
        self.bh_pyk = PyKokkosBarnesHut()

    def read_input_file(self, file):
        self.bh.read_particles_from_file(file)
        self.bh_pyk.read_particles_from_file(file)
        # print(self.bh.particles)

        # print(self.bh_pyk.position)
        # print(self.bh_pyk.mass)
        # print(self.bh_pyk.velocity)


    def run(self, iterations=1):
        self.bh.run(iterations)
        self.bh_pyk.run(iterations)