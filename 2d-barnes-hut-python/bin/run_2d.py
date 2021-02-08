#!/usr/bin/env python3
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from barneshut.implementations import SimpleBarnesHut, ProcessPoolBarnesHut

bh = SimpleBarnesHut()
bh.read_particles_from_file(sys.argv[1])
bh.run(5)
del bh

bh2 = ProcessPoolBarnesHut(5)
bh2.read_particles_from_file(sys.argv[1])
bh2.run(5)