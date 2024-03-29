import threading
from numba import cuda


def test_gpu(idx):
    cuda.select_device(idx)
    dvc = cuda.get_current_device()
    print(f"Thread w/ GPU {idx}.  GPU name/id is {dvc.name} / {dvc.id}")

threads = []
for i in range(2):
    t = threading.Thread(target=test_gpu,args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()