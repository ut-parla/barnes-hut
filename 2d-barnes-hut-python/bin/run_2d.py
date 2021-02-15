#!/usr/bin/env python3
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from barneshut import BarnesHut

fname = sys.argv[1]
n = int(sys.argv[2])
ini_file = sys.argv[3] or None

bh = BarnesHut(ini_file)
bh.read_input_file(fname)
bh.run(n)

# bh2 = ProcessPoolBarnesHut(5)
# bh2.read_particles_from_file(fname)
# bh2.run(n)

# for pt in [1, 10, 100]:
#     bh3 = AsyncBarnesHut()
#     bh3.read_particles_from_file(fname)
#     print(f"Async, {pt} chunk")
#     bh3.run(n, pt)

# for pt in [1, 10, 100]:
#     bh4 = ParlaBarnesHut()
#     bh4.read_particles_from_file(fname)
#     print(f"Parla, {pt} chunk")
#     bh4.run(n, pt)
