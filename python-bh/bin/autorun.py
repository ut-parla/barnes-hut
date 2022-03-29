#!/usr/bin/env python3
from subprocess import run
from time import sleep

cfgs = {
    "singlegpu": "singlegpu.ini",
    "2gpus": "2gpus.ini",
    "4gpus": "4gpus.ini",
    "parla1": "parla1.ini",
    "parla2": "parla2.ini",
    "parla4": "parla4.ini",
}

template_cmd = "./bin/run_2d.py input/n10k.txt 1 1 configs/{ini}"

for name, cfg in cfgs.items():
    cmd = template_cmd.format(ini=cfg)
    fname = f"nbody_out_{name}.txt"

    if name == "parla1":
        cmd = "CUDA_VISIBLE_DEVICES=0 " + cmd
    if name == "parla2":
        cmd = "CUDA_VISIBLE_DEVICES=0,1 " + cmd

    print("running  ", cmd)
    with open(fname+".txt", "w") as outfile:
        run(cmd, shell=True, stdout=outfile)
    sleep(5)
