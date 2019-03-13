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