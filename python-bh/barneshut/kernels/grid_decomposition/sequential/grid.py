import numpy as np
from numba import njit, prange
from barneshut.internals import Config
import barneshut.internals.particle as p
from numpy.lib.recfunctions import structured_to_unstructured as unst

grid_placement_fn = None   
def get_grid_placement_fn():
    global grid_placement_fn
    if grid_placement_fn is None:
        gp = Config.get("sequential", "grid_placement")
        if gp == "np":
            grid_placement_fn = get_grid_placements_numpy
        elif gp == "numba":
            grid_placement_fn = get_grid_placements_numba
        #
    return grid_placement_fn

def get_grid_placements_numpy(particles, min_xy, step, grid_dim):
    """Gets an array of particles (check particle.py for format)
    and finds their place in the grid.
    """
    particles[:, p.gx:p.gy+1] = particles[:, p.px:p.py+1]
    particles[:, p.gx:p.gy+1] = (particles[:, p.gx:p.gy+1] - min_xy) / step
    particles[:, p.gx:p.gy+1] = np.floor(particles[:, p.gx:p.gy+1])

    return particles

# TODO: fix numba, *IF* this is a bottleneck, to see *IF* it is faster
@njit("(float64[:, :], float64[:], float64, int32,)", fastmath=True)
def get_grid_placements_numba(points, min_xy, step, grid_dim):
    # n = points.shape[0]

    # # prange, why not
    # for i in prange(n):
    #     points[i] = (points[i] - min_xy) / step
    #     points[i] = min(int(points[i][0]), grid_dim-1), min(int(points[i][1]), grid_dim-1)

    # return points
    pass
