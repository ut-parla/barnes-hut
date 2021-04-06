# barnes-hut

First the venv must be created and dependencies installed. The Makefile
has a good start on how to do so. Running `make` should work for
Debian based systems. It clones Parla.py and installs it using its setup.py
There is a manual part related to llvm, which is printed during the make.

After the venv is created, run `source ./activate.sh` to activate it.


container dir contains a future container for this workload due to machines having
different CUDA versions, it's not working at this point.
