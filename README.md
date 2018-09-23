# mc-mpi

## Build & Run
Standard CMake project.
```bash
$ git clone https://github.com/lkskstlr/mc-mpi.git
$ cd mc-mpi
$ mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make;
$ mpirun -n 5 ./main 1000000
```
The last line runs on 5 processors with 1000000 particles.

---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.
