import numpy as np
from barneshut.internals import Config

grid_placement_fn = None   
def get_grid_placement_fn():
    global grid_placement_fn
    if grid_placement_fn is None:
        gp = Config.get("sequential", "grid_placement")
        if gp == "np":
            grid_placement_fn = get_grid_placements_numpy
        #
    return grid_placement_fn

def get_grid_placements_numpy(points, min_xy, step, grid_dim):
    points -= min_xy
    points /= step
    # truncate and convert to int
    points = np.trunc(points).astype(int, copy=False)
    points = np.clip(points, 0, grid_dim-1)
    return points


# if we need to sort:
# a = np.array([(1,4,5), (2,1,1), (3,5,1)], dtype='f8, f8, f8')
#   np.argsort(a, order=('f1', 'f2'))
# or a.sort(...)