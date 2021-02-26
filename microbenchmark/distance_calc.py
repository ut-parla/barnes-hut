import numpy as np
from scipy.spatial.distance import pdist
from timeit import default_timer as timer
import statistics

EXPS = 3
WARM = 2

n = 1000
points = np.random.rand(n, 2)
source = np.random.rand(2)

np_times = []
pd_times = []
np2_times = []

for i in range(WARM+EXPS):
    #np naive
    t0 = timer()
    for p1 in points:
        for p2 in points:
            if p1 is not p2:
                np.linalg.norm(p1-p2)        
    t1 = timer()

    if i >= WARM:
        np_times.append(t1-t0)

    #pdist
    t0 = timer()
    pdist(points)
    t1 = timer()

    if i >= WARM:
        pd_times.append(t1-t0)

    #np broadcast
    t0 = timer()
    np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    t1 = timer()

    if i >= WARM:
        np2_times.append(t1-t0)

print(f"numpy: {statistics.mean(np_times)}")
print(f"pdist: {statistics.mean(pd_times)}")
print(f"npbcast: {statistics.mean(np2_times)}")

