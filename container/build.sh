#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cp $DIR/../Makefile .
docker build -f Dockerfile.bh -t hfingler:parla-bh .
