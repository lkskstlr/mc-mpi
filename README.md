# mc-mpi

## Build & Run
Standard CMake project. This assumes you use a Unix system.
```bash
git clone https://github.com/lkskstlr/mc-mpi.git
cd mc-mpi
export CXX=mpicxx
mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make;
mpirun -n 5 ./main 1000000
```
The third line *sets the CXX environment variable* to an mpi compatible compiler. The last line runs on 5 processors with 1000000 particles.


## Tests
A minimal testing script can be found in `test` and can be called by `./test`. A possible output is
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make test_async_comm
Running tests. Should print CORRECT after at most x seconds:
    Test Async Comm ( 1 sec): CORRECT
COMPLETE
```
i.e. one indented line per test. If it says FAILURE (or abort or an error message), the test will internally have failed and noticed the failure. If it prints nothing for more than x sec (here x = 1) there is probably a deadlock and the test didn't pass.

## Software Overview

### Classes
| Class         | Header | @brief      |
|---------------|--------|-------------|
| Partcile etc. |types.hpp  | `real_t` and the Particle struct. |
| UnifDist      |random.hpp | Uniform (0,1] distribution based on mersenne twister. |
| Layer         |layer.hpp  | Representing one cell/layer in the simulation. Physics are implemented here. |
| Worker        |worker.hpp | Manages the control flow of one mpi process. Has members of type `UnifDist`, `Layer`and `AsyncComm`. Each process instantiates exactly one. |
| AsyncComm<T>. |async_comm.hpp| Encapsulation of asynchronous, buffered mpi communication. Internally handles all buffering. Templated. |

### MPI Strategy
Because there is no interaction among the particles, the processes are not synchronized in any way. When enough particles have left the layer of one process, he will send them to his neighbors by a buffered `MPI_Issend` and the `AsyncComm` will handle the buffer. Each process does the following steps
```
simulate -> send -> receive -> receive events -> simulate -> ...
```
The `send`, `receive` and `receive events` return immediately. The events are the communication between master and all other nodes, to determine if the simulation is over.

### Settings
There are different settings that influence behavior. The most important ones are in:
  + `worker.hpp` C-style defines
  + `main.cpp` Settings

---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.


