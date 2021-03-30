from barneshut.internals.config import Config

def nop(*args):
    pass

def get_gravity_kernel():
    
    if "kernel_fn" not in dir(get_gravity_kernel):
        fc = Config.get("sequential", "force_calculation")
        if fc == "vect":
            from . import cpu_vect
            get_gravity_kernel.kernel_fn = cpu_vect.get_kernel_function()
        
        elif fc == "blas":
            from . import cpu_blas
            get_gravity_kernel.kernel_fn = cpu_blas.get_kernel_function()

        elif fc == "numba":
            from . import cpu_numba
            get_gravity_kernel.kernel_fn = cpu_numba.get_kernel_function()

        elif fc == "guvectorize-cpu":
            from . import guvect_cpu
            get_gravity_kernel.kernel_fn = guvect_cpu.get_kernel_function("cpu")

        elif fc == "guvectorize-parallel":
            from . import guvect_cpu
            get_gravity_kernel.kernel_fn = guvect_cpu.get_kernel_function("parallel")

        elif fc == "guvectorize-cuda":
            from . import guvect_cuda
            get_gravity_kernel.kernel_fn = guvect_cuda.get_kernel_function()
            
        elif fc == "nop":
            get_gravity_kernel.kernel_fn = nop

        elif fc == "pykokkos":
            from . import pyk
            get_gravity_kernel.kernel_fn = pyk.get_kernel_function()

    return get_gravity_kernel.kernel_fn