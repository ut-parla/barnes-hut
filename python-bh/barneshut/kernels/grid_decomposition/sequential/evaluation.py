import logging
from itertools import combinations, product
from barneshut.internals import Config
from barneshut.grid_decomposition import Box
from barneshut.kernels.helpers import get_bounding_box, get_neighbor_cells, remove_bottom_left_neighbors

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
        neighbors = get_neighbor_cells(cell, len(grid))
        all_cells = product(    range(n), range(n))
        
        boxes = []
        com_cells = []
        for c in all_cells:
            # for cells that are not neighbors, we need to aggregate COMs into a fake Box
            x,y = c
            if c not in neighbors:
                com_cells.append(grid[x][y])
                logging.debug(f"Cell {c} is not neighbor, appending to COM concatenation")
            # for neighbors, store them so we can do direct interaction
            else:
                boxes.append(grid[x][y])
                logging.debug(f"Cell {c} is neighbor, direct interaction")
        
        coms = Box.from_list_of_boxes(com_cells)
        logging.debug(f"Concatenated COMs have {coms.cloud.n} particles, should have {len(neighbors)}, correct? {coms.cloud.n==len(neighbors)}")
        boxes.append(coms)

        # now we have to do cell <-> box in boxes 
        x,y = cell
        self_leaf = grid[x][y]
        for box in boxes:
            self_leaf.apply_force(box)

def __evaluate_com_concat_dedup(grid):
    n = len(grid)
    # for every box in the grid
    for cell in product(range(n), range(n)):
        neighbors = get_neighbor_cells(cell, len(grid))
        all_cells = product(    range(n), range(n))
        
        boxes = []
        com_cells = []
        for c in all_cells:
            # for cells that are not neighbors, we need to aggregate COMs into a fake Box
            x,y = c
            if c not in neighbors:
                com_cells.append(grid[x][y])
                logging.debug(f"Cell {c} is not neighbor, appending to COM concatenation")
        coms = Box.from_list_of_boxes(com_cells)
        logging.debug(f"Concatenated COMs have {coms.cloud.n} particles, should have {len(neighbors)}, correct? {coms.cloud.n==len(neighbors)}")
        boxes.append(coms)

        # remove boxes that already computed their force to us (this function modifies neighbors list)
        for c in remove_bottom_left_neighbors(cell, neighbors):
            x,y = c
            boxes.append(grid[x][y])
            logging.debug(f"Cell {n} is neighbor, direct interaction")
        
        # now we have to do cell <-> box in boxes 
        x,y = cell
        self_leaf = grid[x][y]
        for box in boxes:
            self_leaf.apply_force(box)