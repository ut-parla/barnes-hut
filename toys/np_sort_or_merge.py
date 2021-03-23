import numpy as np
from timeit import default_timer as timer
import statistics

# For 1M:
#  large qsort: 0.07380389980971813
#  stable sort: 0.06445082982536406
#  small+merge: 0.12443394439760595
#
# For 10M:
#  large qsort: 0.8200506604043767
#  stable sort: 0.7610939522041008
#  small+merge: 1.381166673405096
#
# stable sort it is...


SZ = 10_000_000
TASKS = 4
N_TESTS = 5

def merge_sorted_arrays(a, b):
    m,n = len(a), len(b)
    # Get searchsorted indices
    idx = np.searchsorted(a,b)

    # Offset each searchsorted indices with ranged array to get new positions
    # of b in output array
    b_pos = np.arange(n) + idx

    l = m+n
    mask = np.ones(l,dtype=bool)
    out = np.empty(l,dtype=np.result_type(a,b))
    mask[b_pos] = False
    out[b_pos] = b
    out[mask] = a
    return out

np.random.seed(0)


large_sort_ts = []
stable_sort_ts = []
merges = []
for i in range(N_TESTS):

    large = np.random.randint(0,1000000,(SZ,))
    sortcopy1 = large.copy()
    sortcopy2 = large.copy()

    t0 = timer()
    sortcopy1.sort()
    t1 = timer()
    large_sort_ts.append(t1-t0)
    
    t0 = timer()
    sortcopy2.sort(kind='stable')
    t1 = timer()
    stable_sort_ts.append(t1-t0)

    t0 = timer()
    slices = []
    small_sz = int(SZ/TASKS)
    for i in range(TASKS):
        slices.append( large[i*small_sz:(i+1)*small_sz] )
        #print(f"{i*small_sz} {(i+1)*small_sz}")

    for i in range(TASKS):
        slices[i].sort()

    tasks = int(TASKS/2)
    while tasks != 0:
        for i in range(tasks):
            slices[i] = merge_sorted_arrays(slices[i], slices[tasks+i]  )
        tasks //= 2

    t1 = timer()
    merges.append(t1-t0)

    assert(np.all(np.diff(sortcopy1) >= 0))
    assert(np.all(np.diff(slices[0]) >= 0))

    assert(len(sortcopy1) == len(slices[0]))

print(f"large qsort: {statistics.mean(large_sort_ts)}")
print(f"stable sort: {statistics.mean(stable_sort_ts)}")
print(f"small+merge: {statistics.mean(merges)}")