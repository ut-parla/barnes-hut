# barnes-hut

First the venv must be created and dependencies installed. The Makefile
has a good start on how to do so. Running `make` should work for
Debian based systems. It clones Parla.py and installs it using its setup.py
There is a manual part related to llvm, which is printed during the make.

After the venv is created, run `source ./activate.sh` to activate it.

# sources

It's quite hard to find a Barnes-Hut implementation in python, apparently. Here's a few:

## https://github.com/mikegrudic/pykdgrav/blob/master/setup.py

KD-tree + BH implementation. I have no idea how to run it, though. All python.

## https://github.com/jcummings2/pyoctree/blob/master/octree.py

Octree only. All python.

## https://github.com/ArkUmbra/BarnesHutSimulation

2D BH with quadtree. Has visualization using pygame. All python.

## https://github.com/bacook17/behalf

Octree + BH. Uses a mess of things, like MPI and Cython.
Looks like a good Octree to reuse.

## https://github.com/PWhiddy/Nbody-Gravity/blob/master/BarnzNhutt.cpp

Sanity check for equations