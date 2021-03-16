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
        elif impl == "pykokkos":
            from .implementations.pykokkos import PyKokkosBarnesHut
            bh = PyKokkosBarnesHut()
        elif impl == "singlegpu":
            from .implementations.singlegpu import SingleGPUBarnesHut
            bh = SingleGPUBarnesHut()
        # elif impl == "async":
        #     bh = AsyncBarnesHut()
        #elif impl == "parla":
            #bh = ParlaBarnesHut()
            
        self.bh = bh

    def read_input_file(self, file):
        self.bh.read_particles_from_file(file)

    def run(self, *args, **kwargs):
        self.bh.run(*args, **kwargs)
