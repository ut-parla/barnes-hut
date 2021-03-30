import logging
from itertools import product
from barneshut.internals import Config
from barneshut.grid_decomposition import Box
from barneshut.kernels.helpers import get_bounding_box, get_neighbor_cells, remove_bottom_left_neighbors

#
#  TODO: currently only __evaluate_com_concat_dedup checks if a box is empty. need to implement
# this to other functions if we need to use them because an error occurs for a lot of particles in
# small leaves. This happens because some boxes become empty and there is no checking above these
# functions.
#

eval_fn = None   
def get_evaluation_fn():
    global eval_fn
    if eval_fn is None:
        fn_name = Config.get("sequential", "evaluation")
        if fn_name == "naive":
            eval_fn = __evaluate_naive
        elif fn_name == "com_concat":
            eval_fn = __evaluate_com_concat
        elif fn_name == "com_concat_dedup":
            eval_fn = __evaluate_com_concat_dedup

    return eval_fn

def __evaluate_naive(grid):
    n = len(grid)
    # do all distinct pairs interaction
    cells = product(range(n), range(n))
    pairs = combinations(cells, 2)

    for p1, p2 in pairs:
        l1 = grid[p1[0]][p1[1]]
        l2 = grid[p2[0]][p2[1]]
        l1.apply_force(l2)

    # and all self to self interaction
    for l in range(n):
        leaf = grid[l][l]
        leaf.apply_force(leaf)

def __evaluate_com_concat(grid):
    n = len(grid)
    # for every box in the grid
    for cell in product(range(n), range(n)):
        x,y = cell
        # if box is empty, just skip it
        if grid[x][y].cloud.is_empty():
            print(f"grid {x}/{y} is empty..")
            continue

        self_leaf = grid[x][y]
        neighbors = get_neighbor_cells(cell, len(grid))
        com_cells = []
        for c in product(range(n), range(n)):
            # for cells that are not neighbors, we need to aggregate COMs into a fake Box
            x2,y2 = c
            if grid[x2][y2].cloud.is_empty() or (x==x2 and y==y2):
                continue
            if c not in neighbors:
                com_cells.append(grid[x2][y2])
                #logging.debug(f"Cell {c} is not neighbor, appending to COM concatenation")
            # for neighbors, store them so we can do direct interaction
            else:
                print(f"direct interaction of {x}/{y}  -> {x2}/{y2}")
                self_leaf.apply_force(grid[x2][y2], update_other=False)
                
        coms = Box.from_list_of_boxes(com_cells, is_COMs=True)
        self_leaf.apply_force(coms, update_other=False)

        # we also need to interact with ourself
        self_leaf.apply_force(self_leaf,  update_other=False)


def __evaluate_com_concat_dedup(grid):
    n = len(grid)
    # for every box in the grid
    for cell in product(range(n), range(n)):
        x,y = cell
        # if box is empty, just skip it
        if grid[x][y].cloud.is_empty():
            print(f"grid {x}/{y} is empty..")
            continue
        self_leaf = grid[x][y]
        neighbors = get_neighbor_cells(cell, len(grid))
        
        com_cells = []
        for c in product(range(n), range(n)):
            # for cells that are not neighbors, we need to aggregate COMs into a fake Box
            x2,y2 = c
            # if box is empty, just skip it
            if grid[x2][y2].cloud.is_empty() or (x==x2 and y==y2):
                continue
            if c not in neighbors:
                com_cells.append(grid[x2][y2])
        if len(com_cells) != 0:
            coms = Box.from_list_of_boxes(com_cells, is_COMs=True)
            #logging.debug(f'''Concatenated COMs have {coms.cloud.n} particles, 
            #        should have {n*n-len(neighbors)}, correct? {coms.cloud.n==n*n-len(neighbors)}''')
            self_leaf.apply_force(coms, update_other=False)

        # remove boxes that already computed their force to us (this function modifies neighbors list)
        for c in remove_bottom_left_neighbors(cell, neighbors):
            x2,y2 = c
            # if box is empty, just skip it
            if grid[x2][y2].cloud.is_empty():
                continue
            self_leaf.apply_force(grid[x2][y2], update_other=True)

        # we also need to interact with ourself
        self_leaf.apply_force(self_leaf, update_other=False)
