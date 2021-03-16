from numpy.lib.recfunctions import structured_to_unstructured as unst
from numba import cuda, float64
from math import floor, fabs

# i hate this hardcoded stuff. numpy
# has a thingy to get the idx given
# name, maybe someday...
_px, _py = 0, 1
_mas = 2
_ax, _ay = 5, 6
_gx, _gy = 7, 8

def place_particles(particles, min_xy, step, grid_dim):

    d_grid_box_count = cuda.device_array((grid_dim,grid_dim),  dtype=np.int)
    # TODO: zero d_grid_box_count
    d_grid_box_cumm = cuda.device_array((grid_dim,grid_dim),  dtype=np.int)
    d_grid_box_count_cumm_copy = cuda.device_array((grid_dim,grid_dim),  dtype=np.int)

    d_particles = cuda.to_device(unst(particles))
    d_particles_ordered = cuda.device_array_like(unst(particles))

    #TODO: calc grid, thread size

    # dealloc what we dont need
    d_particles = None
    d_grid_box_count_cumm_copy = None


    #2d (x,y, mass) array for each box in the grid
    d_COMs = cuda.device_array((3, grid_dim,grid_dim), dtype=np.float)




#TODO: check orientation of ndarray, we might have to transpose for performance
@cuda.jit
def g_place_particles(particles, particles_ordered, min_xy, step, grid_dim, grid_box_count, grid_box_cumm, grid_box_cumm_copy):
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

    while tid < particles.size:
        particles[tid, gx] = (particles[tid, _px] - min_x) / step
        particles[tid, gx] = min(floor(particles[tid, _gx]), grid_dim-1)
        particles[tid, gy] = (particles[tid, _py] - min_y) / step
        particles[tid, gy] = min(floor(particles[tid, _gy]), grid_dim-1)
        x = int(particles[tid, _gx])
        y = int(particles[tid, _gy])
        # add 1 to the index x,y
        cuda.atomic.add(grid_box_count, (x,y), 1)
        # go around
        tid += tsize

    cuda.syncthreads()

    # I don't think this is worth launching a kernel for since 
    # grid is usually pretty small
    if tid == 0:
        acc = 0
        for i in range(grid_dim):
            for j in range(grid_dim):
                acc += grid_box_count[i,j]
                # add to accumulate arrays
                grid_box_cumm[i,j]      = acc
                grid_box_cumm_copy[i,j] = acc

    # after this point we have cumulative box count, which means the last
    # element of grid_box_count is == n

    cuda.syncthreads()

    tid = cuda.grid(1)
    while tid < particles.size:
        x = particles[tid, _gx]
        y = particles[tid, _gy]
        # this is the idx this particle will be put into
        new_idx = cuda.atomic.add(grid_box_cumm_copy, (x,y), -1) - 1
        # copy it there
        particles_ordered[new_idx] = particles[tid]
        
        tid += tsize

@cuda.jit(device=True)
def d_previous_box_count(grid_box_count, gx, gy, grid_dim):
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
    return grid_box_count[x, y]

@cuda.jit
def g_summarize(particles, grid_box_count, grid_box_cumm, grid_dim, COMs):
    """Write something better. I just want it to work.
    Launch one thread per box in the grid.
    """
    my_x, my_y = cuda.grid(2)
    end = grid_box_count[x,y]
    start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)

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

    COMs[0][my_x][my_y] = acc_x / M
    COMs[1][my_x][my_y] = acc_x / M

@cuda.jit(device=True)
def d_is_neighbor(gx, gy, gx2, gy2):
    if fabs(gx-gx2) <= 1 and fabs(gy-gy2) <= 1:
        return 1
    else: 
        return 0

@cuda.jit
def d_self_self_grav(particles, start, end, G):
    n = end-start
    tid = cuda.threadIdx.x
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

@cuda.jit
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

@cuda.jit
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
def g_evaluate(particles, grid_dim, grid_box_cumm, COMs, G):
    """
    Let's do the easy thing: launch one block per box in the grid, launch threads
    equal to the max # of particles in a box
    """
    my_x = cuda.blockIdx.x
    my_y = cuda.blockIdx.y

    end = grid_box_cumm[my_x, my_y]
    start = d_previous_box_count(grid_box_cumm, my_x, my_y, grid_dim)

    for gx in range(grid_dim): 
        for gy in range(grid_dim):
            # self to self
            if gx == my_x and gy == my_y:
                d_self_self_grav(particles, start, end, G)
            # neighbors, direct p2p interaction
            elif d_is_neighbor(my_x, my_y, gx, gy):
                other_end = grid_box_cumm[gx, gy]
                other_start = d_previous_box_count(grid_box_cumm, gx, gy, grid_dim)
                d_self_other_grav(particles, start, end, other_start, other_end, G)
            # not neighbor, use COM
            else:
                d_self_COM_grav(particles, start, end, COMs, gx, gy, G)