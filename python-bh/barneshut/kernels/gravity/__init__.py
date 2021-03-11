from barneshut.internals.config import Config

def nop(*args):
    pass

fn = None   

def get_gravity_kernel():
    global fn

    if fn is None:
        fc = Config.get("general", "force_calculation")
        if fc == "vect":
            from . import cpu_vect
            fn = cpu_vect.get_kernel_function()

        elif fc == "blas":
            from . import cpu_blas
            fn = cpu_blas.get_kernel_function()

        elif fc == "numba":
            from . import cpu_numba
            fn = cpu_numba.get_kernel_function()

        elif fc == "guvectorize-cpu":
            from . import guvect_cpu
            fn = guvect_cpu.get_kernel_function("cpu")

        elif fc == "guvectorize-parallel":
            from . import guvect_cpu
            fn = guvect_cpu.get_kernel_function("parallel")

        elif fc == "guvectorize-cuda":
            from . import guvect_cuda
            fn = guvect_cuda.get_kernel_function()
            
        elif fc == "nop":
            fn = nop

    return fn