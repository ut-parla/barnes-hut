import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from barneshut.simplebarneshut import SimpleBarnesHut

bh = SimpleBarnesHut()
bh.read_particles_from_file(sys.argv[1])
bh.run(5)