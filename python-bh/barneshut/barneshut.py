from barneshut.internals.config import Config
from .implementations.sequential import SequentialBarnesHut

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

    def read_input_file(self, file):
        self.bh.read_particles_from_file(file)

    def run(self, iterations=1):
        self.bh.run(iterations)