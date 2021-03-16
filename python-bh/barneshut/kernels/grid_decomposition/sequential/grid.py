import numpy as np
from numba import njit, prange
from barneshut.internals import Config
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
    For a particle p, we use p['px'], p['py'] and set
    p['gx'], p['gy']
    """
    particles[['gx','gy']] = particles[['px','py']]

    # hard coding the index after unst seems to be the only way of 
    # not copying and actually having a reference
    grid_coords = unst(particles, copy=False)[:, 7:9]
    grid_coords = (grid_coords - min_xy) / step
    # clipping makes truncate unnecessary
    #print(f"before clip {grid_coords}")
    grid_coords = np.trunc(grid_coords)
    grid_coords = np.clip(grid_coords, 0, grid_dim-1)

    unst(particles, copy=False)[:, 7:9] = grid_coords

    #print(f"parts: {particles}")
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
