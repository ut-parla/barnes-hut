import numpy as np
from math import ceil
from numba import cuda, float64, njit
from itertools import product
import barneshut.internals.particle as p
from barneshut.internals import Cloud
from parla.tasks import *
from parla.function_decorators import *
from parla.cuda import *
from parla.cpu import *

from barneshut.kernels.helpers import get_neighbor_cells, remove_bottom_left_neighbors
from barneshut.kernels.grid_decomposition.gpu.grid import *


@specialized
@njit(fastmath=True)
def p_place_particles(particles, grid_cumm, min_xy, grid_dim, step):
    min_x, min_y = min_xy[0], min_xy[1]
    for pi in range(particles.shape[0]):
        particles[pi, p.gx] = (particles[pi, p.px] - min_x) / step
        particles[pi, p.gx] = min(floor(particles[pi, p.gx]), grid_dim-1)
        particles[pi, p.gy] = (particles[pi, p.py] - min_y) / step
        particles[pi, p.gy] = min(floor(particles[pi, p.gy]), grid_dim-1)

        gx, gy = int(particles[pi, p.gx]), int(particles[pi, p.gy])
        grid_cumm[gx, gy] += 1

@p_place_particles.variant(gpu)
def p_place_particles_gpu(particles, grid_cumm, min_xy, grid_dim, step):
    threads_per_block = float(Config.get("cuda", "threads_per_block"))
    blocks = ceil(particles.shape[0] / threads_per_block)
    threads = threads_per_block
    g_place_particles[blocks, threads](particles, min_xy, step, grid_dim, grid_cumm)

def previous_box(grid, gx, gy, grid_dim):
    if gx == 0 and gy == 0:
        return 0
    if gx == 0:
        return grid[grid_dim-1, gy-1]
    else:
        return grid[gx-1, gy]

@specialized
@njit(fastmath=True)
def p_summarize_boxes(particle_slice, box_list, offset, grid_ranges, grid_dim, COMs):
    for box in box_list:
        x, y = box
        start, end = grid_ranges[x, y]

        M, acc_x, acc_y = .0, .0, .0
        for pt in range(start-offset, end-offset):
            mass = particle_slice[pt, p.mass]
            acc_x += particle_slice[pt, p.px] * mass
            acc_y += particle_slice[pt, p.py] * mass
            M += mass
        if M != .0:
            COMs[x, y, 0] = acc_x / M
            COMs[x, y, 1] = acc_y / M
            COMs[x, y, 2] = M

@p_summarize_boxes.variant(gpu)
def p_summarize_boxes_gpu(particle_slice, box_list, offset, grid_ranges, grid_dim, COMs):
    blocks = 1
    threads = len(box_list)
    g_summarize_w_ranges[blocks, threads](particle_slice, box_list, offset, grid_ranges, grid_dim, COMs)

@cuda.jit
def g_summarize_w_ranges(particles, box_list, offset, grid_ranges, grid_dim, COMs):
    tid = cuda.grid(1)
    my_x, my_y = box_list[tid]
    start, end = grid_ranges[my_x, my_y]

    M = .0
    acc_x = .0
    acc_y = .0

    #print("calculating COM of {}/{}. Start/end: {} - {}".format(my_x, my_y, start, end))
    for i in range(start, end):
        pi = i-offset
        mass = particles[pi, p.mass]
        acc_x += particles[pi, p.px] * mass
        acc_y += particles[pi, p.py] * mass
        M += mass
    if M != .0:
        COMs[my_x, my_y, 0] = acc_x / M
        COMs[my_x, my_y, 1] = acc_y / M
        COMs[my_x, my_y, 2] = M

@specialized
def p_evaluate(_particles, my_boxes, grid, _grid_ranges, COMs, G, grid_dim):
    for box in my_boxes:
        x, y = box
        if grid[(x,y)].is_empty():
            continue

        neighbors = get_neighbor_cells((x,y), grid_dim)
        com_boxes = []
        for ox in range(grid_dim):
            for oy in range(grid_dim):
                other_box = (ox, oy)
                # if box is empty, just skip it
                if grid[other_box].is_empty() or (x==ox and y==oy):
                    continue
                if other_box not in neighbors:
                    com_boxes.append(other_box)

        concatenated_COMs = np.empty((len(com_boxes), 3))
        for i, cb in enumerate(com_boxes):
            cx, cy = cb
            concatenated_COMs[i] = COMs[cx, cy]

        CCOMS = Cloud.from_slice(concatenated_COMs)
        # interact with non-neighbors
        grid[(x,y)].apply_force(CCOMS, update_other=False)

        # interact with neighbors
        for n in neighbors:
            ox, oy = n
            grid[(x,y)].apply_force(grid[(ox,oy)], update_other=False)

        # we also need to interact with ourself
        grid[(x,y)].apply_force(grid[(x,y)], update_other=False)

@p_evaluate.variant(gpu)
def p_evaluate_gpu(particles, my_boxes, grid, grid_ranges, COMs, G, grid_dim):
    threads_per_block = float(Config.get("cuda", "threads_per_block"))
    cn = set()
    for box_xy in my_boxes:
        cn |= set(get_neighbor_cells(tuple(box_xy), grid_dim))
    tp_boxes = [(x,y) for x,y in my_boxes]
    cn -= set(tp_boxes)
    #print(f"Close neighbors of {tp_boxes} : {cn}")

    # now we need to copy those boxes
    n_close_neighbors = len(cn)
    total = 0
    cn_particles = np.empty((1, 3), dtype=np.float64)
    cn_ranges = np.zeros((grid_dim, grid_dim, 2), dtype=np.int32)
    if n_close_neighbors > 0:
        for x, y in cn:
            l = grid_ranges[x, y, 1] - grid_ranges[x, y, 0]
            cn_ranges[x, y] = total, total+l
            total += l                 
        cn_particles = np.empty((total, 3), dtype=np.float64)
        for x, y in cn:
            start, end = cn_ranges[x, y]
            ostart, oend = grid_ranges[x, y]
            cn_particles[start:end] = particles[ostart:oend, p.px:p.mass+1]

    fb_x, fb_y = my_boxes[0]
    lb_x, lb_y = my_boxes[-1]
    offset = grid_ranges[fb_x, fb_y, 0]
    start = grid_ranges[fb_x, fb_y, 0]
    end = grid_ranges[lb_x, lb_y, 1]
    my_particles = particles[start:end]

    pblocks = ceil((end-start)/threads_per_block)
    blocks = (pblocks, len(my_boxes))
    threads = threads_per_block

    g_evaluate_parla_multigpu[blocks, threads](my_particles, my_boxes, grid_ranges, offset, grid_dim, 
                COMs, cn_ranges, cn_particles, G)

    return my_particles

@cuda.jit
def g_evaluate_parla_multigpu(particles, my_boxes, grid_ranges, offset, grid_dim, COMs, cn_ranges, cn_particles, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    box_idx = cuda.blockIdx.y
    tid = cuda.grid(1)
    gx, gy = my_boxes[box_idx]
    start, end = grid_ranges[gx, gy]
    start -= offset
    end -= offset
    n = end-start

    if n > 0 and tid < n:
        # for each other grid cell
        for other_gx in range(grid_dim): 
            for other_gy in range(grid_dim):
                # self to self
                if gx == other_gx and gy == other_gy:
                    pass
                    d_self_self_grav_mgpu(particles, start, end, G)
                # neighbors, direct p2p interaction
                elif d_is_neighbor(other_gx, other_gy, gx, gy):
                    ns = cn_ranges[other_gx, other_gy, 0]
                    ne = cn_ranges[other_gx, other_gy, 1]
                    # if both are zero, it's a cell from our GPU
                    #print("ns ne ", ns, " ", ne)
                    if ns == 0 and ne == 0:
                        ostart, oend = grid_ranges[other_gx, other_gy]
                        ostart -= offset
                        oend -= offset
                        #print("eval neighbor same gpu other, grids: ", gx, "-", gy, "  ", other_gx, "-", other_gy, "  start/end ",  start, "-", end, "   ", ostart, "-", oend)
                        d_self_other_grav(particles, start, end, ostart, oend, G)
                     # if not, it's from another GPU, so we need to use the indices
                    else:
                        #print("grid range ", other_gx, "-", other_gy, "  = ", ns, " ", ne)
                        d_self_other_mgpu_neighbor_grav(particles, start, end, cn_particles, ns, ne, G)
                ## not neighbor, use COM
                else:
                    d_self_COM_grav(particles, start, end, COMs, other_gx, other_gy, G)
