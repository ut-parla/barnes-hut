from .barneshut.sequential  import SequentialBarnesHut
#from .barneshut.processpool import ProcessPoolBarnesHut
#from .barneshut.asyncio     import AsyncBarnesHut
from .barneshut.parla       import ParlaBarnesHut
from barneshut.internals.config import Config

class BarnesHut:

    def __init__(self, ini_file):
        Config.read_file(ini_file)
        impl = Config.get("general", "implementation", fallback="sequential")

        bh = None
        if impl == "sequential":
            bh = SequentialBarnesHut()
        # elif impl == "process":
        #     bh = ProcessPoolBarnesHut()
        # elif impl == "async":
        #     bh = AsyncBarnesHut()
        elif impl == "parla":
            bh = ParlaBarnesHut()
        print(f"Using {impl} implementation")
        self.bh = bh

    def read_input_file(self, file):
        self.bh.read_particles_from_file(file)

    def run(self, iterations=1):
        self.bh.run(iterations)