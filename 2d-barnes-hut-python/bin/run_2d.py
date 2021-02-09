#!/usr/bin/env python3
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from barneshut.implementations import SimpleBarnesHut, ProcessPoolBarnesHut, AsyncBarnesHut, ParlaBarnesHut

fname = sys.argv[1]
n = int(sys.argv[2])

bh = SimpleBarnesHut()
bh.read_particles_from_file(fname)
bh.run(n)

# bh2 = ProcessPoolBarnesHut(5)
# bh2.read_particles_from_file(fname)
# bh2.run(n)

#bh3 = AsyncBarnesHut()
#bh3.read_particles_from_file(fname)
#bh3.run(n)

bh4 = ParlaBarnesHut()
bh4.read_particles_from_file(fname)
bh4.run(n)
