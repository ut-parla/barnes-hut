from numpy.lib.recfunctions import structured_to_unstructured as unst
from numba import cuda, float64
from math import floor, fabs

CUDA_DEBUG = False

# i hate this hardcoded stuff. numpy
# has a thingy to get the idx given
# name, maybe someday...
_px, _py = 0, 1
_mas = 2
_vx, _vy = 3, 4
_ax, _ay = 5, 6
_gx, _gy = 7, 8

@cuda.jit(device=True)
def copy_point(src, src_idx, dest, dest_idx):
    for i in range(9):
        dest[dest_idx][i] = src[src_idx][i]

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
        particles[pidx, _gx] = (particles[pidx, _px] - min_x) / step
        particles[pidx, _gx] = min(floor(particles[pidx, _gx]), grid_dim-1)
        particles[pidx, _gy] = (particles[pidx, _py] - min_y) / step
        particles[pidx, _gy] = min(floor(particles[pidx, _gy]), grid_dim-1)
        x = int(particles[pidx, _gx])
        y = int(particles[pidx, _gy])

        #print(f"grid: {grid_dim}  {particles[pidx, _gx]}/{particles[pidx, _gy]}  = {x}/{y}")

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
    for j in range(grid_dim):
        for i in range(grid_dim):
            acc += grid_box_count[i,j]
            # add to accumulate arrays
            grid_box_count[i,j] = acc
            grid_box_cumm[i,j]  = acc

            if CUDA_DEBUG:
                print("Cumm: {}/{}  =  {}".format(i, j, grid_box_cumm[i,j]))
            #print("Cumm: ", i, " ", j, " = ", grid_box_cumm[i,j])
    # after this point we have cumulative box count, which means the last
    # element of grid_box_count is == n

@cuda.jit
def g_sort_particles(particles, particles_ordered, grid_box_count):
    tid   = cuda.grid(1)
    tsize = cuda.gridsize(1)
    n_particles = particles.shape[0]

    pidx = tid
    while pidx < n_particles:
        x = int(particles[pidx, _gx])
        y = int(particles[pidx, _gy])
        #print("tid ", tid, " x ", x, " y ", y)
        # this is the idx this particle will be put into
        new_idx = cuda.atomic.add(grid_box_count, (x,y), -1) - 1
        #if CUDA_DEBUG:
            #print("copy from {} to {}".format(pidx, new_idx))
        # copy it there
        copy_point(particles, pidx, particles_ordered, new_idx)
        pidx += tsize

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

@cuda.jit
def g_summarize(particles, grid_box_cumm, grid_dim, COMs):
    """Write something better. I just want it to work.
    Launch one thread per box in the grid.
    """
    my_x, my_y = cuda.grid(2)

    print(f"x,y {my_x} {my_y}")
    if my_x < grid_dim and my_y < grid_dim:
        start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
        end = grid_box_cumm[my_x, my_y]
        COMs[my_x][my_y][0] = 0
        COMs[my_x][my_y][1] = 0
        COMs[my_x][my_y][2] = 0
        print("start/end ", start," ", end)
        if start != end:  #requred for multi gpu        
            M = .0
            acc_x = .0
            acc_y = .0

            print("calculating COM of {}/{}. Start/end: {} - {}".format(my_x, my_y, start, end))
            for i in range(start, end):
                px = particles[i, _px]
                py = particles[i, _py]
                mass = particles[i, _mas]
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

@cuda.jit(device=True)
def d_is_neighbor(gx, gy, gx2, gy2):
    if fabs(gx-gx2) <= 1 and fabs(gy-gy2) <= 1:
        return 1
    else: 
        return 0

@cuda.jit(device=True)
def d_self_self_grav(particles, start, end, G):    
    n = end-start
    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    pid = start+tid

    # we are particle tid
    if tid < n:
        my_x = particles[pid, _px]
        my_y = particles[pid, _py]
        my_mass = particles[pid, _mas]
        for i in range(start, end):
            # skip if both are us
            if pid != i:
                ox, oy = particles[i, _px], particles[i, _py]
                xdif, ydif = my_x-ox, my_y-oy
                dist = (xdif*xdif + ydif*ydif)**0.5
                f = (G * my_mass * particles[i, _mas]) / (dist*dist)
                # update only ourselves since the other will calc to us
                particles[pid, _ax] -= (f * xdif / my_mass)
                particles[pid, _ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_other_grav(particles, start, end, other_start, other_end, G):
    n = end-start
    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    # offset our id so we get particles between [start,end)
    pid = start+tid

    # we are particle tid
    if tid < n:
        my_x = particles[pid, _px]
        my_y = particles[pid, _py]
        my_mass = particles[pid, _mas]
        for i in range(other_start, other_end):
            ox, oy = particles[i, _px], particles[i, _py]
            xdif, ydif = my_x-ox, my_y-oy
            dist = (xdif*xdif + ydif*ydif)**0.5
            f = (G * my_mass * particles[i, _mas]) / (dist*dist)
            # update only ourselves since the other will calc to us
            particles[pid, _ax] -= (f * xdif / my_mass)
            particles[pid, _ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_COM_grav(particles, start, end, COMs, cx, cy, G):
    n = end-start
    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    # offset our id so we get particles between [start,end)
    pid = start+tid

    # we are particle tid
    if tid < n:
        my_x, my_y = particles[pid, _px], particles[pid, _py]
        my_mass = particles[pid, _mas]
        com_x, com_y = COMs[cx, cy, 0], COMs[cx, cy, 1]
        com_mass = COMs[cx, cy, 2]

        xdif, ydif = my_x-com_x, my_y-com_y
        dist = (xdif*xdif + ydif*ydif)**0.5
        f = (G * my_mass * com_mass) / (dist*dist)
        # update only ourselves since the other will calc to us
        particles[pid, _ax] -= (f * xdif / my_mass)
        particles[pid, _ay] -= (f * ydif / my_mass)

@cuda.jit
def g_evaluate_boxes(particles, grid_dim, grid_box_cumm, COMs, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    my_y = int(cuda.blockIdx.x / grid_dim)
    my_x = cuda.blockIdx.x - (my_y*grid_dim)
    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    #print("grix {}/{}  tid{}".format(my_x, my_y, tid))
    start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
    end = grid_box_cumm[my_x, my_y]
    n = end-start

    #if my_x == 0 and my_y == 0:
    if tid < n:
        for gx in range(grid_dim): 
            for gy in range(grid_dim):
                # self to self
                if gx == my_x and gy == my_y:
                    d_self_self_grav(particles, start, end, G)
                # neighbors, direct p2p interaction
                elif d_is_neighbor(my_x, my_y, gx, gy):
                    other_end = grid_box_cumm[gx, gy]
                    other_start = d_previous_box_count(grid_box_cumm, gx, gy, grid_dim)
                    #print("{}  eval  {} - {}".format(tid, other_start, other_end))
                    d_self_other_grav(particles, start, end, other_start, other_end, G)
                # not neighbor, use COM
                else:
                    d_self_COM_grav(particles, start, end, COMs, gx, gy, G)

@cuda.jit
def g_recalculate_box_cumm(particles, grid_box_cumm, grid_dim):
    tid = cuda.grid(1)
    tsize = cuda.gridsize(1)
    n_particles = particles.shape[0]
    
    # since we can't zero an array from host, let's do it here real quick
    max_gb = grid_dim*grid_dim
    if tid < max_gb:
        row = int(tid / grid_dim)
        col = tid - (row*grid_dim)
        grid_box_cumm[row, col] = 0

    cuda.syncthreads()

    pidx = tid
    while pidx < n_particles:
        x = int(particles[pidx, _gx])
        y = int(particles[pidx, _gy])

        if not CUDA_DEBUG:
            cuda.atomic.add(grid_box_cumm, (x,y), 1)
        else:
            old = cuda.atomic.add(grid_box_cumm, (x,y), 1)
            print("recalc grid {}/{} = {}".format(x, y, old))
        
        # go around
        pidx += tsize

    cuda.syncthreads()

    if tid == 0:
        acc = 0
        for j in range(grid_dim):
            for i in range(grid_dim):
                acc += grid_box_cumm[i,j]
                # add to accumulate arrays
                grid_box_cumm[i,j]  = acc

@cuda.jit(device=True)
def d_self_other_mgpu_neighbor_grav(particles, start, end, neighbors, ns, ne, G):
    n = end-start
    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    # offset our id so we get particles between [start,end)
    pid = start+tid

    # we are particle tid
    if tid < n:
        my_x = particles[pid, _px]
        my_y = particles[pid, _py]
        my_mass = particles[pid, _mas]
        for i in range(ns, ne):
            ox, oy = neighbors[i, _px], neighbors[i, _py]
            xdif, ydif = my_x-ox, my_y-oy
            dist = (xdif*xdif + ydif*ydif)**0.5
            f = (G * my_mass * neighbors[i, _mas]) / (dist*dist)
            # update only ourselves since the other will calc to us
            particles[pid, _ax] -= (f * xdif / my_mass)
            particles[pid, _ay] -= (f * ydif / my_mass)

@cuda.jit
def g_evaluate_boxes_multigpu(particles, grid_dim, grid_box_cumm, COMs, G, neighbors, neighbors_indices, cells, ncells):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    cell_idx = cuda.blockIdx.x
    my_y = cells[cell_idx][0]
    my_x = cells[cell_idx][1]

    tid = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.x
    
    start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
    end = grid_box_cumm[my_x, my_y]
    n = end-start

    if start != end and tid < n:
        # for each other grid cell
        for gx in range(grid_dim): 
            for gy in range(grid_dim):
                # self to self
                if gx == my_x and gy == my_y:
                    d_self_self_grav(particles, start, end, G)
                # neighbors, direct p2p interaction
                elif d_is_neighbor(my_x, my_y, gx, gy):
                    ns, ne = neighbors_indices[gx, gy]
                    # if both are zero, it's a cell from our GPU
                    if ns == 0 and ne == 0:
                        other_end = grid_box_cumm[gx, gy]
                        other_start = d_previous_box_count(grid_box_cumm, gx, gy, grid_dim)
                        #print("{}  eval  {} - {}".format(tid, other_start, other_end))
                        d_self_other_grav(particles, start, end, other_start, other_end, G)
                    # if not, it's from another GPU, so we need to use the indices
                    else:
                        d_self_other_mgpu_neighbor_grav(particles, start, end, neighbors, ns, ne, G)
            
                # not neighbor, use COM
                else:
                    d_self_COM_grav(particles, start, end, COMs, gx, gy, G)


@cuda.jit
def g_tick_particles(particles, tick):
    tid   = cuda.grid(1)
    tsize = cuda.gridsize(1)
    n_particles = particles.shape[0]

    # find where each particle belongs in the grid
    pidx = tid
    while pidx < n_particles:
        particles[pidx, _vx] += particles[pidx, _ax] * tick
        particles[pidx, _vy] += particles[pidx, _ay] * tick

        particles[pidx, _ax] = 0
        particles[pidx, _ay] = 0

        particles[pidx, _px] += particles[pidx, _vx] * tick
        particles[pidx, _py] += particles[pidx, _vy] * tick

        pidx += tsize