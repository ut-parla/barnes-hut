#!/usr/bin/env python3
from subprocess import run
from time import sleep


cfgs = {
    # "singlegpu": "singlegpu.ini",
    # "2gpus": "2gpus.ini",
    # "4gpus": "4gpus.ini",
    # "parla1": "parla1.ini",
    # "parla2": "parla2.ini",
    # "parla4": "parla4.ini",

    #"parla1_eager": "parla1_eager.ini",
    #"parla2_eager": "parla2_eager.ini",
    #"parla4_eager": "parla4_eager.ini",

    "parla1_eager_sched": "parla1_eager_sched.ini",
    "parla2_eager_sched": "parla2_eager_sched.ini",
    "parla4_eager_sched": "parla4_eager_sched.ini",
}
}

template_cmd = "./bin/run_2d.py input/nbody-10M.txt 1 1 configs/{ini}"

for name, cfg in cfgs.items():
    cmd = template_cmd.format(ini=cfg)
    fname = f"nbody_out_{name}"

    if name == "parla1":
        cmd = "CUDA_VISIBLE_DEVICES=0 " + cmd
    if name == "parla2":
        cmd = "CUDA_VISIBLE_DEVICES=0,1 " + cmd

    print("running  ", cmd)
    with open(fname+".dat", "w") as outfile:
        run(cmd, shell=True, stdout=outfile)
    sleep(5)
