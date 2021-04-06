# 2d quadtree Barnes-Hut


## Setup

Assuming you already have the venv and it is activated, generate the inputs:

`./bin/generate_sample_inputs.sh`

which will generate inputs with up to 100k particles. To generate the 1M and 10M inputs, 
which take a while, run the commands manually:

`python3 bin/gen_input.py normal 1000000 input/n1M.txt`
`python3 bin/gen_input.py normal 10000000 input/n10M.txt`


## Running

First you have to choose the implementation and its options.
For this you can either modify `configs/default.ini` or
copy it to a different file and modify it.
The file should be self explanatory.

Then you can run it:

`./bin/run_2d.py input/n10k.txt 1 1 configs/default.ini`

where the first argument is input file, second is warmup rounds,
third is measuring runs and fourth is the config file to be used.
There are optional arguments:
`--debug` will print debug messages.
`--check` will measure accuracy, but takes a long time for larger inputs.