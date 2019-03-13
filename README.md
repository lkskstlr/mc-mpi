# mc-mpi [![Build Status](https://travis-ci.org/lkskstlr/mc-mpi.svg?branch=inf518)](https://travis-ci.org/lkskstlr/mc-mpi)

## Build & Test
Standard CMake project. This assumes you use a Unix system.
```shell-session
git clone https://github.com/lkskstlr/mc-mpi.git
cd mc-mpi && mkdir build && cd build && cmake .. && make -j
make test
```
To build with CUDA support use `cmake .. -DCUDA_ENABLED=ON` and make sure to either have `g++-4.8` in your `PATH` or set `CUDAHOSTCXX` to a [supported compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers).

## Run
After building use
```shell-session
mpirun -n 5 ./main ../config.yaml rma
```
to run. The last argument can be `sync`, `async` or `rma` for the three implemented communication strategies.

The results will be saved in `build/out` which should not be touched by the user. If you want to save this specific run, call `./save_run` from the build directory and the run will be saved to `../saved_runs/<timestamp>.tar.gz`. Also the `latest` symlink in `saved_runs` will be set for convenience.

---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.
