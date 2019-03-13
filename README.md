# INF 518 - Project
[![Build Status](https://travis-ci.org/lkskstlr/mc-mpi.svg?branch=inf518)](https://travis-ci.org/lkskstlr/mc-mpi)

## Build & Test
Standard CMake project. This assumes you use a Unix system.
```shell-session
mkdir build && cd build && cmake .. && make -j
make test
```


## Scope
The following files are exclusive (with associated headers) for this project:
+ `*_comm.cpp`
+ `*_comm-impl.cpp`
+ `layer.cpp` (without CUDA and OpenMP)
+ `particle_*_comm.cpp`
+ `stats.cpp`
+ `test_async_serialization.cpp`
+ `test_comm.cpp`
+ `test_layer.cpp`
+ `test_rma_serialization.cpp`
+ `test_state_comm.cpp`
+ `timer.cpp`
+ `worker.cpp`
+ `worker_async.cpp`
+ `worker_rma.cpp`
+ `yaml_dumper.cpp`
+ `yaml_loader.cpp`