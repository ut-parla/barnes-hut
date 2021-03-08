#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
INPUT="$DIR/../input"

mkdir -p $INPUT
python3 $DIR/gen_input.py normal 100    $INPUT/n100.txt
python3 $DIR/gen_input.py normal 1000   $INPUT/n1k.txt
python3 $DIR/gen_input.py normal 10000  $INPUT/n10k.txt
python3 $DIR/gen_input.py normal 100000 $INPUT/n100k.txt