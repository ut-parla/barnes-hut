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

parla_logger = logging.getLogger('parla')
parla_logger.setLevel(logging.WARNING)

# import entry point
from timer import Timer
from barneshut import BarnesHut

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to input file")
    parser.add_argument('nwarmups', help="How many runs to execute as warmup", type=int)
    parser.add_argument('nruns', help="Number of different runs to execute", type=int)
    parser.add_argument('configfile', help="Path to config file")
    parser.add_argument('--debug', help="Turn debug on", action="store_true")
    parser.add_argument('--check', help="Check accuracy after every round", action="store_true")
    args = parser.parse_args()

    nruns   = args.nruns
    nwarmups= args.nwarmups
    fname   = args.input_file
    cfgfile = args.configfile
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.disable(logging.CRITICAL)

    check = args.check

    bh = BarnesHut(cfgfile)
    bh.read_input_file(fname)

    for _ in range(nwarmups):
        bh.run(check_accuracy=check)
    print("warmup done")
    Timer.reset()
    for _ in range(nruns):
        bh.run(check_accuracy=check)
    Timer.print()