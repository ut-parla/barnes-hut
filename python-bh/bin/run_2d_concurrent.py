#!/usr/bin/env python3
import os, sys
import argparse
from time import sleep

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
import threading

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to input file")
    parser.add_argument('nwarmups', help="How many runs to execute as warmup", type=int)
    parser.add_argument('nruns', help="Number of different runs to execute", type=int)
    parser.add_argument('concurrent', help="Number of different concurrent apps to run", type=int)
    parser.add_argument('configfile', help="Path to config file")
    parser.add_argument('--debug', help="Turn debug on", action="store_true")
    parser.add_argument('--check', help="Check accuracy after every round", action="store_true")
    args = parser.parse_args()

    nruns   = args.nruns
    nwarmups= args.nwarmups
    concurrent = args.concurrent
    fname   = args.input_file
    cfgfile = args.configfile
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.disable(logging.CRITICAL)

    check = args.check

    bhs = []
    threads = []
    for j in range(concurrent):
        bh = BarnesHut(cfgfile)
        bhs.append(bh)
        t = threading.Thread(target=bh.read_input_file, args=(fname,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    threads = []
    for _ in range(nwarmups):
        for i, bh in enumerate(bhs):
            t = threading.Thread(target=bh.run, kwargs={"check_accuracy": check})
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    
    print("warmup done")
    sleep(2)
    Timer.reset()
    threads = []
    for _ in range(nruns):
        for i, bh in enumerate(bhs):
            t = threading.Thread(target=bh.run, kwargs={"check_accuracy": check}) #, "suffix": str(i)})
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        sleep(2)
    Timer.print()