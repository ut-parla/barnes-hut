import numpy as np
from numpy import sqrt, power, ceil
from itertools import product

def next_perfect_square(n):
    if n % sqrt(n) != 0:
        return power(ceil(sqrt(n)), 2)
    return n

def get_bounding_box(points):
    """ Get bounding box coordinates around all particles.
        Returns the bottom left and top right corner coordinates, making
        sure that it is a square.
    """
    max_x, max_y = np.max(points, axis=0)[:2]
    min_x, min_y = np.min(points, axis=0)[:2]

    x_edge, y_edge = max_x - min_x, max_y - min_y 
    if x_edge >= y_edge:
        max_y += (x_edge - y_edge)
    else:
        max_x += (y_edge - x_edge)

    assert (max_x-min_x)==(max_y-min_y)
    return (min_x, min_y), (max_x, max_y)

def get_neighbor_cells(point, grid_dim):
    x, y = point
    x_start = x-1 if x>0 else x
    x_end = x+1 if x < grid_dim-1 else x
    y_start = y-1 if y>0 else y
    y_end = y+1 if y < grid_dim-1 else y

    cells = list(product(range(x_start, x_end+1), range(y_start, y_end+1)))
    cells.remove(point)
    return cells

    #print(f"\n\nneighbors of {point}")
    #for c in cells:
    #    print(c)

def remove_bottom_left_neighbors(point, neighbors):
    x,y = point
    pts = [(x-1,y-1), (x-1,y), (x,y-1), (x+1,y-1)]
    
    for pt in pts:
        try:
            neighbors.remove(pt)
        except:
            pass
    return neighbors