from numpy.lib.recfunctions import structured_to_unstructured as unst
from numba import cuda, float64
from math import floor, fabs

CUDA_DEBUG = True

# i hate this hardcoded stuff. numpy
# has a thingy to get the idx given
# name, maybe someday...
_px, _py = 0, 1
_mas = 2
_ax, _ay = 5, 6
_gx, _gy = 7, 8

@cuda.jit(device=True)
def copy_point(src, src_idx, dest, dest_idx):
    for i in range(9):
        dest[dest_idx][i] = src[src_idx][i] 

#TODO: check orientation of ndarray, we might have to transpose for performance
@cuda.jit
def g_place_particles(particles, particles_ordered, min_xy, step, grid_dim, grid_box_count, grid_box_cumm, max_box):
    """
    Args:
        particles: ndarray, array of particles
        particles_ordered: ndarray, same shape as particles.  TODO: do this in-place.
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

    # since we can't zero an array from host, let's do it here real quick
    max_gb = grid_dim*grid_dim
    if tid < max_gb:
        row = int(tid / grid_dim)
        col = tid - (row*grid_dim)
        grid_box_count[row, col] = 0

    cuda.syncthreads()

    pidx = tid
    while pidx < n_particles:
        particles[pidx, _gx] = (particles[pidx, _px] - min_x) / step
        particles[pidx, _gx] = min(floor(particles[pidx, _gx]), grid_dim-1)
        particles[pidx, _gy] = (particles[pidx, _py] - min_y) / step
        particles[pidx, _gy] = min(floor(particles[pidx, _gy]), grid_dim-1)
        x = int(particles[pidx, _gx])
        y = int(particles[pidx, _gy])
        # add 1 to the index x,y

        if not CUDA_DEBUG:
            cuda.atomic.add(grid_box_count, (x,y), 1)
        else:
            old = cuda.atomic.add(grid_box_count, (x,y), 1)
            print("grid {}/{} = {}".format(x, y, old))
        
        # go around
        pidx += tsize

    cuda.syncthreads()

    # I don't think this is worth launching a kernel for since 
    # grid is usually pretty small
    if tid == 0:
        acc = 0
        max_box = 0
        for j in range(grid_dim):
            for i in range(grid_dim):
                if grid_box_count[i,j] > max_box:
                    max_box = grid_box_count[i,j]
                acc += grid_box_count[i,j]
                # add to accumulate arrays
                grid_box_count[i,j] = acc
                grid_box_cumm[i,j]  = acc

                if CUDA_DEBUG:
                    print("Cumm: {}/{}  =  {}".format(i, j, grid_box_cumm[i,j]))
            
    # after this point we have cumulative box count, which means the last
    # element of grid_box_count is == n

    cuda.syncthreads()

    pidx = tid
    while pidx < n_particles:
        x = int(particles[pidx, _gx])
        y = int(particles[pidx, _gy])
        # this is the idx this particle will be put into
        new_idx = cuda.atomic.sub(grid_box_count, (x,y), 1) - 1
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
    my_x = cuda.blockIdx.x
    my_ = 

    
    
    start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
    end = grid_box_cumm[my_x, my_y]
    M = .0
    acc_x = .0
    acc_y = .0

    #print("calculating COM of {}/{}. Start/end: {} - {}".format(my_x, my_y, start, end))

    for i in range(start, end):
        px = particles[i, _px]
        py = particles[i, _py]
        mass = particles[i, _mas]
        acc_x += px * mass
        acc_y += py * mass
        M += mass

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
    tid = cuda.threadIdx.x
    n = end-start

    # we are particle tid
    if tid < n:
        my_x = particles[tid, _px]
        my_y = particles[tid, _py]
        my_mass = particles[tid, _mas]

        for i in range(start, end):
            # skip if both are us
            if tid != i:
                ox, oy = particles[i, _px], particles[i, _py]
                xdif, ydif = my_x-ox, my_y-oy
                dist = (xdif*xdif + ydif*ydif)**0.5
                f = (G * my_mass * particles[i, _mas]) / (dist*dist)
                # update only ourselves since the other will calc to us
                particles[tid, _ax] -= (f * xdif / my_mass)
                particles[tid, _ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_other_grav(particles, start, end, other_start, other_end, G):
    n = end-start
    tid = cuda.threadIdx.x
    # we are particle tid
    if tid < n:
        my_x = particles[tid, _px]
        my_y = particles[tid, _py]
        my_mass = particles[tid, _mas]

        for i in range(other_start, other_end):
            ox, oy = particles[i, _px], particles[i, _py]
            xdif, ydif = my_x-ox, my_y-oy
            dist = (xdif*xdif + ydif*ydif)**0.5
            f = (G * my_mass * particles[i, _mas]) / (dist*dist)
            # update only ourselves since the other will calc to us
            particles[tid, _ax] -= (f * xdif / my_mass)
            particles[tid, _ay] -= (f * ydif / my_mass)

@cuda.jit(device=True)
def d_self_COM_grav(particles, start, end, COMs, cx, cy, G):
    n = end-start
    tid = cuda.threadIdx.x
    # we are particle tid
    if tid < n:
        my_x, my_y = particles[tid, _px], particles[tid, _py]
        my_mass = particles[tid, _mas]
        com_x, com_y = COMs[0, cx, cy], [1, cx, cy]
        com_mass = COMs[2, cx, cy]

        xdif, ydif = my_x-com_x, my_y-com_y
        dist = (xdif*xdif + ydif*ydif)**0.5
        f = (G * my_mass * com_mass) / (dist*dist)
        # update only ourselves since the other will calc to us
        particles[tid, _ax] -= (f * xdif / my_mass)
        particles[tid, _ay] -= (f * ydif / my_mass)

@cuda.jit
def g_evaluate_boxes(particles, grid_dim, grid_box_cumm, COMs, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """

    my_y = cuda.blockIdx.x
    my_y = int(my_x / grid_dim)
    my_x = cuda.blockIdx.x - (my_y*grid_dim)

    tid = cuda.blockIdx.y * cuda.blockDim.y

    print("grix {}/{}  tid{}".format(my_x, my_y, tid))



    # start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)
    # end = grid_box_cumm[my_x, my_y]

    # if my_x == 0 and my_y == 0:

    #     for gx in range(grid_dim): 
    #         for gy in range(grid_dim):
    #             # self to self
    #             if gx == my_x and gy == my_y:
    #                 d_self_self_grav(particles, start, end, G)
    #             # # neighbors, direct p2p interaction
    #             # elif d_is_neighbor(my_x, my_y, gx, gy):
    #             #     other_end = grid_box_cumm[gx, gy]
    #             #     other_start = d_previous_box_count(grid_box_cumm, gx, gy, grid_dim)
    #             #     d_self_other_grav(particles, start, end, other_start, other_end, G)
    #             # # not neighbor, use COM
    #             # else:
    #             #     d_self_COM_grav(particles, start, end, COMs, gx, gy, G)