#!/usr/bin/env python3
import os, sys
import argparse

# add path witchery so we can import all modules correctly
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# setup logging
import logging

# avoid warnings by using OMP
from numba import config, threading_layer
config.THREADING_LAYER = 'omp'
# set numba's logging otherwise it fills the screen
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# import entry point
from barneshut import BarnesHut

def run(cfgfile, fname, nrounds, check):
    # run the thingy
    bh = BarnesHut(cfgfile)
    bh.read_input_file(fname)
    bh.run(nrounds, check_accuracy=check)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to input file")
    parser.add_argument('nrounds', help="Number of rounds to run", type=int)
    parser.add_argument('configfile', help="Path to config file")
    parser.add_argument('--debug', help="Turn debug on", action="store_true")
    parser.add_argument('--check', help="Check accuracy after every round", action="store_true")
    args = parser.parse_args()

    fname   = args.input_file
    nrounds = args.nrounds
    cfgfile = args.configfile
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    check = args.check

    run(cfgfile, fname, nrounds, check)