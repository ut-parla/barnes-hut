import numpy as np
import cupy as cp
from math import ceil
from numba import cuda, float64, njit
from itertools import product
import barneshut.internals.particle as p
from barneshut.internals import Cloud
from barneshut.internals.config import Config
from parla.tasks import *
from parla.function_decorators import *
from parla.cuda import *
from parla.cpu import *
from timer import Timer

from barneshut.kernels.helpers import get_neighbor_cells, remove_bottom_left_neighbors
from barneshut.kernels.grid_decomposition.gpu.grid import *

def stream_cupy_to_numba(cp_stream):
    '''
    Notes:
        1. The lifetime of the returned Numba stream should be as long as the CuPy one,
           which handles the deallocation of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same CUDA context as the
           CuPy one.
        3. The implementation here closely follows that of cuda.stream() in Numba.
    '''
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)
    finalizer = None  # let CuPy handle its lifetime, not Numba

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(weakref.proxy(ctx), handle, finalizer)

    return nb_stream


MAX_X_BLOCKS = 65535
#max is 65535

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
    threads_per_block = int(Config.get("cuda", "threads_per_block"))
    blocks = ceil(particles.shape[0] / threads_per_block)
    blocks = min(blocks, MAX_X_BLOCKS)
    threads = threads_per_block

    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    g_place_particles[blocks, threads, nb_stream](particles, min_xy, step, grid_dim, grid_cumm)
    nb_stream.synchronize()


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
    threads_per_block = int(Config.get("cuda", "threads_per_block"))
    blocks = ceil(len(box_list) / threads_per_block)
    threads = min(threads_per_block, len(box_list))
    #print(f"blocks {blocks} threads {threads}  box_list: {box_list}")
    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    g_summarize_w_ranges[blocks, threads, nb_stream](particle_slice, box_list, offset, grid_ranges, grid_dim, COMs)
    nb_stream.synchronize()

# @specialized
# def p_evaluate(_particles, my_boxes, grid, _grid_ranges, COMs, G, grid_dim):
#     concatenated_COMs = np.empty((grid_dim*grid_dim, 3))
#     for box in my_boxes:
#         with Timer.get_handle("box"):
#             x, y = box
#             if grid[(x,y)].is_empty():
#                 continue

#             with Timer.get_handle("ccomappend_neighbors"):

#                 neighbors = get_neighbor_cells((x,y), grid_dim)
#                 com_count = 0
#                 for ox in range(grid_dim):
#                     for oy in range(grid_dim):
#                         other_box = (ox, oy)
#                         # if box is empty, just skip it
#                         if grid[other_box].is_empty() or (x==ox and y==oy):
#                             continue
#                         if other_box not in neighbors:
#                             concatenated_COMs[com_count] = COMs[ox, oy]
#                             com_count += 1
#                         # interact with neighbors
#                         else:
#                             grid[(x,y)].apply_force(grid[(ox,oy)], update_other=False)

#             with Timer.get_handle("ccom_self"):
#                 CCOMS = Cloud.from_slice(concatenated_COMs[:com_count])
#                 # interact with non-neighbors
#                 grid[(x,y)].apply_force(CCOMS, update_other=False)

#                 # we also need to interact with ourself
#                 grid[(x,y)].apply_force(grid[(x,y)], update_other=False)

#@p_evaluate.variant(gpu)
#def p_evaluate_gpu(particles, my_boxes, _grid, grid_ranges, COMs, G, grid_dim):
def p_evaluate_eager(particles, my_boxes, _grid, grid_ranges, COMs, G, grid_dim, slices, is_parray):
    threads_per_block = int(Config.get("cuda", "threads_per_block"))
    cn = set()
    for box_xy in my_boxes:
        cn |= set(get_neighbor_cells(tuple(box_xy), grid_dim))
    tp_boxes = [(x,y) for x,y in my_boxes]
    cn -= set(tp_boxes)
    #print(f"Close neighbors of {tp_boxes} : {cn}")

    # now we need to copy those boxes
    n_close_neighbors = len(cn)
    total = 0
    cn_particles = cp.empty((1, 3), dtype=np.float64)
    cn_ranges = np.zeros((grid_dim, grid_dim, 2), dtype=np.int32)
    if n_close_neighbors > 0:
        for x, y in cn:
            l = grid_ranges[x, y, 1] - grid_ranges[x, y, 0]
            cn_ranges[x, y] = total, total+l
            total += l
        cn_particles = cp.empty((total, 3), dtype=np.float64)
        for x, y in cn:
            start, end = cn_ranges[x, y]
            ostart, oend = grid_ranges[x, y]

            if not is_parray:
                cn_particles[start:end] = particles[ostart:oend, p.px:p.mass+1].array
            else:
                crosses_border = False
                for i in range(len(slices)-1):
                    if ostart < slices[i][1] and oend > slices[i+1][0]:
                        crosses_border = True
                        sl1 = slices[i]
                        sl2 = slices[i+1]
                if not crosses_border:
                    cn_particles[start:end] = particles[ostart:oend, p.px:p.mass+1].array
                else:
                    ncopy1 = sl1[1]-ostart
                    cn_particles[start:start+ncopy1] = particles[ostart:ostart+ncopy1, p.px:p.mass+1].array
                    ncopy2 = oend-ostart-ncopy1
                    cn_particles[start+ncopy1:start+ncopy1+ncopy2] = particles[ostart+ncopy1:ostart+ncopy1+ncopy2, p.px:p.mass+1].array


    fb_x, fb_y = my_boxes[0]
    lb_x, lb_y = my_boxes[-1]
    offset = grid_ranges[fb_x, fb_y, 0]
    start = grid_ranges[fb_x, fb_y, 0]
    end = grid_ranges[lb_x, lb_y, 1]
    my_particles = particles[start:end].array

    pblocks = ceil((end-start)/threads_per_block)
    gblocks = min(len(my_boxes), MAX_X_BLOCKS)
    blocks = (pblocks, gblocks)
    threads = threads_per_block

    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    g_evaluate_parla_multigpu[blocks, threads, nb_stream](my_particles, my_boxes, grid_ranges, offset, grid_dim,
                COMs, cn_ranges, cn_particles, G)
    nb_stream.synchronize()
    cuda.synchronize()
    return my_particles


def p_evaluate(particles, my_boxes, _grid, grid_ranges, COMs, G, grid_dim, slices, is_parray):
    threads_per_block = int(Config.get("cuda", "threads_per_block"))
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
    gblocks = min(len(my_boxes), MAX_X_BLOCKS)
    blocks = (pblocks, gblocks)
    threads = threads_per_block
    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    g_evaluate_parla_multigpu[blocks, threads, nb_stream](my_particles, my_boxes, grid_ranges, offset, grid_dim,
                COMs, cn_ranges, cn_particles, G)
    nb_stream.synchronize()
    cuda.synchronize()
    return my_particles


@specialized
@njit(fastmath=True)
def p_timestep(particles, tick):
    for pi in range(particles.shape[0]):
        particles[pi, p.vx] += particles[pi, p.ax] * tick
        particles[pi, p.vy] += particles[pi, p.ay] * tick
        particles[pi, p.ax] = 0
        particles[pi, p.ay] = 0
        particles[pi, p.px] += particles[pi, p.vx] * tick
        particles[pi, p.py] += particles[pi, p.vy] * tick

@p_timestep.variant(gpu)
def p_timestep(particles, tick):
    threads_per_block = int(Config.get("cuda", "threads_per_block"))
    blocks = ceil(particles.shape[0] / threads_per_block)
    blocks = min(blocks, MAX_X_BLOCKS)
    threads = threads_per_block
    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    g_tick_particles[blocks, threads, nb_stream](particles, tick)
    nb_stream.synchronize()
    cuda.synchronize()
