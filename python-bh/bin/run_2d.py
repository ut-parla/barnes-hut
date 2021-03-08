#!/usr/bin/env python3
import os, sys

# add path witchery so we can import all modules correctly
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# avoid warnings by using OMP
from numba import config, threading_layer
config.THREADING_LAYER = 'omp'

# setup logging
import logging
logging.basicConfig(level=logging.DEBUG)

# import entry point
from barneshut import BarnesHut


# TODO: use argparse
fname = sys.argv[1]
n = int(sys.argv[2])
ini_file = sys.argv[3] or None


# run the thingy
bh = BarnesHut(ini_file)
bh.read_input_file(fname)
bh.run(n)