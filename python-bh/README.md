# 2d quadtree Barnes-Hut


## TODO:

- [ ] Use GPU/Numba
- [ ] Add cost based partitioning
- [ ] Improve/parallelize tree building 
- [ ] Fix multi process implementation
- [x] Make a tree node hold more than one particle
- [x] Set print particle positions as optional flag
- [x] Chunk particles to reduce task-set size. (result: not good, at least for parla)
- [x] Cleanup pygame
- [x] Input generator, save input to file for use in benchmark
- [x] Use some task abstraction, like Python's async
- [x] Add Parla