# barnes-hut

run source ./activate.sh to create/activate the venv, will automatically install dependencies.
It has a dependency on Parla, so it must be installed. I did it by activating the venv, cloning the
Parla repo (https://github.com/ut-parla/Parla.py) and running python3 setup.py install.
Parla is in the requirements.txt but it probably doesn't work.


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