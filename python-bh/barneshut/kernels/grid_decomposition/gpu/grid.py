from numba import cuda, float64
from math import floor, fabs
import barneshut.internals.particle as p

CUDA_DEBUG = False


#######################################################################
#
#   Single GPU kernels
#
#######################################################################

#TODO: check orientation of ndarray, we might have to transpose for performance
@cuda.jit
def g_place_particles(particles, min_xy, step, grid_dim, grid_box_count):
    """
    Args:
        particles: ndarray, array of particles
        min_xy: (min_x, min_y), bottom left point of bounding box
        step: size of each edge of the grid
        grid_dim: amount of boxes in the grid
        grid_box_count: a (grid_dim x grid_dim) matrix with all elements = 0
        grid_box_count_copy: copy of grid_box_count since we modify it
    """
    tid   = cuda.grid(1)
    tsize = cuda.gridsize(1)
    min_x, min_y = min_xy[0], min_xy[1]
    n_particles = particles.shape[0]

    # find where each particle belongs in the grid
    pidx = tid
    while pidx < n_particles:
        # lets zero acceleration here
        #particles[pidx, _ax] = .0
        #particles[pidx, _ay] = .0
        particles[pidx, p.gx] = (particles[pidx, p.px] - min_x) / step
        particles[pidx, p.gx] = min(floor(particles[pidx, p.gx]), grid_dim-1)
        particles[pidx, p.gy] = (particles[pidx, p.py] - min_y) / step
        particles[pidx, p.gy] = min(floor(particles[pidx, p.gy]), grid_dim-1)
        x = int(particles[pidx, p.gx])
        y = int(particles[pidx, p.gy])

        # add 1 to the index x,y
        if not CUDA_DEBUG:
            cuda.atomic.add(grid_box_count, (x,y), 1)
        else:
            old = cuda.atomic.add(grid_box_count, (x,y), 1)
            print("grid {}/{} = {}".format(x, y, old))
        # go around
        pidx += tsize

@cuda.jit
def g_calculate_box_cumm(grid_dim, grid_box_count, grid_box_cumm):
    acc = 0
    for i in range(grid_dim):
        for j in range(grid_dim):
            acc += grid_box_count[j,i]
            # add to accumulate arrays
            grid_box_count[j,i] = acc
            grid_box_cumm[j,i]  = acc

@cuda.jit
def g_sort_particles(particles, particles_ordered, grid_box_count):
    tid   = cuda.grid(1)
    tsize = cuda.gridsize(1)
    n_particles = particles.shape[0]

    pidx = tid
    while pidx < n_particles:
        x = int(particles[pidx, p.gx])
        y = int(particles[pidx, p.gy])
        # this is the idx this particle will be put into
        new_idx = cuda.atomic.add(grid_box_count, (x,y), -1) - 1
        # copy it there
        copy_point(particles, pidx, particles_ordered, new_idx)
        pidx += tsize

@cuda.jit
def g_summarize(particles, grid_box_cumm, grid_dim, COMs):
    """Write something better. I just want it to work.
    Launch one thread per box in the grid.
    """
    my_x, my_y = cuda.grid(2)

    if my_x < grid_dim and my_y < grid_dim:
        start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
        end = grid_box_cumm[my_x, my_y]

        #print(f"box ", my_x, "/", my_y, "  start/end ", start, " ", end)

        COMs[my_x][my_y][0] = 0
        COMs[my_x][my_y][1] = 0
        COMs[my_x][my_y][2] = 0
        if start != end:  #requred for multi gpu        
            M = .0
            acc_x = .0
            acc_y = .0

            #print("calculating COM of {}/{}. Start/end: {} - {}".format(my_x, my_y, start, end))
            for i in range(start, end):
                px = particles[i, p.px]
                py = particles[i, p.py]
                mass = particles[i, p.mass]
                acc_x += px * mass
                acc_y += py * mass
                M += mass
            if M != .0:
                COMs[my_x][my_y][0] = acc_x / M
                COMs[my_x][my_y][1] = acc_y / M
                COMs[my_x][my_y][2] = M

            if CUDA_DEBUG:
                print("COM of {}/{} is  {}/{}  with mass {}".format(my_x, my_y,
                        COMs[my_x][my_y][0], COMs[my_x][my_y][1], COMs[my_x][my_y][2]))

@cuda.jit
def g_evaluate_boxes(particles, grid_dim, grid_box_cumm, COMs, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    current_g = cuda.blockIdx.y
    wraps_grid = cuda.gridDim.y
    total_boxes = grid_dim*grid_dim
    
    while current_g < total_boxes:
        my_y = int(current_g / grid_dim)
        my_x = current_g - (my_y*grid_dim)

        start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
        end = grid_box_cumm[my_x, my_y]
        n = end-start
        tid = cuda.grid(1)
        tsize = cuda.gridsize(1)
        wraps = 0

        pidx = tid + (wraps*tsize)
        while pidx < n:
            for gx in range(grid_dim): 
                for gy in range(grid_dim):
                    # self to self
                    if gx == my_x and gy == my_y:
                        d_self_self_grav(particles, start, end, G, wraps)
                    # neighbors, direct p2p interaction
                    elif d_is_neighbor(my_x, my_y, gx, gy):
                        other_end = grid_box_cumm[gx, gy]
                        other_start = d_previous_box_count(grid_box_cumm, gx, gy, grid_dim)
                        d_self_other_grav(particles, start, end, other_start, other_end, G, wraps)
                    # not neighbor, use COM
                    else:
                        d_self_COM_grav(particles, start, end, COMs, gx, gy, G, wraps)
            #wrap around particles
            wraps += 1
            pidx = tid + (wraps*tsize)
        #wrap around grid boxes
        current_g += wraps_grid


@cuda.jit
def g_tick_particles(particles, tick):
    tid   = cuda.grid(1)
    tsize = cuda.gridsize(1)
    n_particles = particles.shape[0]

    # find where each particle belongs in the grid
    pidx = tid
    while pidx < n_particles:
        particles[pidx, p.vx] += particles[pidx, p.ax] * tick
        particles[pidx, p.vy] += particles[pidx, p.ay] * tick
        particles[pidx, p.ax] = 0
        particles[pidx, p.ay] = 0
        particles[pidx, p.px] += particles[pidx, p.vx] * tick
        particles[pidx, p.py] += particles[pidx, p.vy] * tick
        pidx += tsize

#######################################################################
#
#   Common kernels to all that use GPUs
#
#######################################################################

@cuda.jit(device=True)
def copy_point(src, src_idx, dest, dest_idx):
    for i in range(p.nfields):
        dest[dest_idx][i] = src[src_idx][i]

@cuda.jit(device=True)
def d_is_neighbor(gx, gy, gx2, gy2):
    if fabs(gx-gx2) <= 1 and fabs(gy-gy2) <= 1:
        return 1
    else: 
        return 0

@cuda.jit(device=True)
def d_previous_box_count(grid_box, gx, gy, grid_dim):
    if gx == 0 and gy == 0:
        return 0
    x, y = 0, 0
    # if leftmost col, need to go one row down
    if gx == 0:
        x = grid_dim - 1
        y = gy - 1
    # easy case, just go left
    else:
        x = gx - 1
        y = gy

    #print("prev of {}/{} is {}/{}\nid {}.  Current cumm {}  previous: {}".format(gx, gy, x, y, cuda.grid(2), grid_box_count[gx,gy], grid_box_count[x,y] ))
    return grid_box[x, y]

@cuda.jit(device=True)
def d_self_self_grav(particles, start, end, G, wraps):    
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    pidx = tid + (wraps*tsize)
    pid = start+pidx

    my_x = particles[pid, p.px]
    my_y = particles[pid, p.py]
    my_mass = particles[pid, p.mass]
    for i in range(start, end):
        # skip if both are us
        if pid != i:
            ox, oy = particles[i, p.px], particles[i, p.py]
            xdif, ydif = my_x-ox, my_y-oy
            dist = (xdif*xdif + ydif*ydif)**0.5
            f = (G * my_mass * particles[i, p.mass]) / (dist*dist)
            # update only ourselves since the other will calc to us
            particles[pid, p.ax] -= (f * xdif / my_mass)
            particles[pid, p.ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_other_grav(particles, start, end, other_start, other_end, G, wraps):
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    pidx = tid + (wraps*tsize)
    pid = start+pidx

    my_x = particles[pid, p.px]
    my_y = particles[pid, p.py]
    my_mass = particles[pid, p.mass]
    for i in range(other_start, other_end):
        ox, oy = particles[i, p.px], particles[i, p.py]
        xdif, ydif = my_x-ox, my_y-oy
        dist = (xdif*xdif + ydif*ydif)**0.5
        f = (G * my_mass * particles[i, p.mass]) / (dist*dist)
        # update only ourselves since the other will calc to us
        particles[pid, p.ax] -= (f * xdif / my_mass)
        particles[pid, p.ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_COM_grav(particles, start, end, COMs, cx, cy, G, wraps):
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    pidx = tid + (wraps*tsize)
    pid = start+pidx

    my_x, my_y = particles[pid, p.px], particles[pid, p.py]
    my_mass = particles[pid, p.mass]
    com_x, com_y = COMs[cx, cy, 0], COMs[cx, cy, 1]
    com_mass = COMs[cx, cy, 2]

    xdif, ydif = my_x-com_x, my_y-com_y
    dist = (xdif*xdif + ydif*ydif)**0.5
    f = (G * my_mass * com_mass) / (dist*dist)
    # update only ourselves since the other will calc to us
    particles[pid, p.ax] -= (f * xdif / my_mass)
    particles[pid, p.ay] -= (f * ydif / my_mass)

#######################################################################
#
#   Multi GPU and Parla kernels
#
#######################################################################

@cuda.jit(device=True)
def d_self_other_mgpu_neighbor_grav(particles, start, end, neighbors, ns, ne, G, wraps):
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    pidx = tid + (wraps*tsize)
    pid = start+pidx

    my_x = particles[pid, p.px]
    my_y = particles[pid, p.py]
    my_mass = particles[pid, p.mass]
    for i in range(ns, ne):
        ox, oy = neighbors[i, p.px], neighbors[i, p.py]
        xdif, ydif = my_x-ox, my_y-oy
        dist = (xdif*xdif + ydif*ydif)**0.5
        f = (G * my_mass * neighbors[i, p.mass]) / (dist*dist)
        # update only ourselves since the other will calc to us
        particles[pid, p.ax] -= (f * xdif / my_mass)
        particles[pid, p.ay] -= (f * ydif / my_mass)
        #print("remote_neighbor updating  ", pid, " id ", particles[pid, p.pid])

@cuda.jit(device=True)
def d_self_self_grav_mgpu(particles, start, end, G, wraps):    
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    pidx = tid + (wraps*tsize)
    pid = start+pidx

    # we are particle tid
    my_x = particles[pid, p.px]
    my_y = particles[pid, p.py]
    my_mass = particles[pid, p.mass]
    for i in range(start, end):
        # skip if both are us
        if pid != i:
            #print("interact ", pid, "->", i)
            ox, oy = particles[i, p.px], particles[i, p.py]
            xdif, ydif = my_x-ox, my_y-oy
            dist = (xdif*xdif + ydif*ydif)**0.5
            f = (G * my_mass * particles[i, p.mass]) / (dist*dist)
            # update only ourselves since the other will calc to us
            particles[pid, p.ax] -= (f * xdif / my_mass)
            particles[pid, p.ay] -= (f * ydif / my_mass)

@cuda.jit
def g_summarize_w_ranges(particles, box_list, offset, grid_ranges, grid_dim, COMs):
    tid = cuda.grid(1)
    if tid < len(box_list):
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

@cuda.jit
def g_evaluate_parla_multigpu(particles, my_boxes, grid_ranges, offset, grid_dim, COMs, cn_ranges, cn_particles, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    wraps_grid = cuda.gridDim.y
    box_idx = cuda.blockIdx.y

    while box_idx < len(my_boxes):
        gx, gy = my_boxes[box_idx]
        box_idx += wraps_grid

        start, end = grid_ranges[gx, gy]
        start -= offset
        end -= offset
        n = end-start

        tid = cuda.grid(1)
        tsize = cuda.gridsize(1)
        wraps = 0

        pidx = tid + (wraps*tsize)
        while pidx < n:
            # for each other grid cell
            for other_gx in range(grid_dim): 
                for other_gy in range(grid_dim):
                    # self to self
                    if gx == other_gx and gy == other_gy:
                        d_self_self_grav_mgpu(particles, start, end, G, wraps)
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
                            d_self_other_grav(particles, start, end, ostart, oend, G, wraps)
                        # if not, it's from another GPU, so we need to use the indices
                        else:
                            #print("grid range ", other_gx, "-", other_gy, "  = ", ns, " ", ne)
                            d_self_other_mgpu_neighbor_grav(particles, start, end, cn_particles, ns, ne, G, wraps)
                    ## not neighbor, use COM
                    else:
                        d_self_COM_grav(particles, start, end, COMs, other_gx, other_gy, G, wraps)
            wraps += 1
            pidx = tid + (wraps*tsize)
