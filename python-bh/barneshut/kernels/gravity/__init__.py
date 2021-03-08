from barneshut.internals.config import Config
from . import cpu_blas
from . import cpu_vect
from . import guvect_cpu
from . import guvect_cuda

fn = None   

def get_gravity_kernel():
    global fn

    if fn is None:
        fc = Config.get("general", "force_calculation")
        if fc == "vect":
            fn = cpu_vect.get_kernel_function()
        elif fc == "blas":
            fn = cpu_blas.get_kernel_function()
        elif fc == "guvectorize-cpu":
            fn = guvect_cpu.get_kernel_function("cpu")
        elif fc == "guvectorize-parallel":
            fn = guvect_cpu.get_kernel_function("parallel")
        elif fc == "guvectorize-cuda":
            fn = guvect_cuda.get_kernel_function()

    return fn