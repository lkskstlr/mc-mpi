# INF 560 Project
[![Build Status](https://travis-ci.org/lkskstlr/mc-mpi.svg?branch=inf560)](https://travis-ci.org/lkskstlr/mc-mpi)

Ezequiel Morcillo Rosso, Lukas Koestler


## Build & Test
Standard CMake project. This assumes you use a Unix system.
```shell-session
git clone https://github.com/lkskstlr/mc-mpi.git && cd mc-mpi
git checkout inf560
source set_env.sh
mkdir build && cd build && cmake .. -DCUDA_ENABLED=On && make -j
make test
```


## Scope
The following files (and associated headers) exclusively belong to the INF560 project:
+ `worker_sync.cpp`
+ `culayer.cu`
+ `culayer_kernel.cu`
+ `gpu_detect.cu`
+ `gpu_detect.cpp`
+ `test_curandom.cu`
+ `test_culayer.cu`
+ `test_gpu_errcheck`
+ `layer.cpp` (OpenMP and CUDA sections)