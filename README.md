# mc-mpi

## Build & Run
Standard CMake project. This assumes you use a Unix system.
```shell-session
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
| Partcile etc. |types.hpp  | `real_t` and the `Particle` struct. |
| UnifDist      |random.hpp | Uniform (0,1] distribution based on mersenne twister. |
| Layer         |layer.hpp  | Representing one cell/layer in the simulation. Physics are implemented here. |
| Worker        |worker.hpp | Manages the control flow of one mpi process. Has members of type `UnifDist`, `Layer`and `AsyncComm`. Each process instantiates exactly one worker. |
| AsyncComm&lt;T&gt; |async_comm.hpp| Encapsulation of asynchronous, buffered mpi communication. Internally handles all buffering. Templated. |

### MPI Strategy
Because there is no interaction among the particles, the processes are not synchronized in any way. When the process has completed a certain number of simulation steps, he will send the particles which left the layer to his neighbors by a buffered `MPI_Issend` and the `AsyncComm` will handle the buffer. Each process does the following steps
```
simulate -> send -> receive -> receive events -> simulate -> ...
```
The `send`, `receive` and `receive events` return immediately. The events are the communication between master and all other nodes, to determine if the simulation is over.

### Settings
There are different settings that influence behavior. The most important ones are in:
  + `worker.hpp` C-style defines
  + `main.cpp` Settings


### Logging
A minimal logging implementation can be found in `include/logging.h`. The basic usage is as follows:
```cpp
MPI_Init(NULL, NULL);
int world_size, world_rank;

MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

MCMPI_DEBUG_INIT(world_rank)
MCMPI_DEBUG("My world rank is %d", world_rank)

MCMPI_DEBUG_STOP()
MPI_Finalize();
```

The logs are written to `logs/<world_rank>.txt` where the user has to create the folder `logs` in the directory where the binary is run, e.g. in `build`. An example can be found in `toy_examples/logging.c`. The logging is global in all files that include `logging.h` but local per mpi processor. If one compiles with `-DNDEBUG` (included in `-DCMAKE_BUILD_TYPE=Release`) the logging disappears without a trace and doesn't have any runtime costs. This can be checked by `cd toy_examples; make test;`.

### Timing
A minimal timing class is provided in `include/timer.hpp` and can be used as follows:
```cpp
Timer timer;
auto id = timer.start(Timer::Tag::Computation);
int a = 1;
for (int i = 0; i < 1000; ++i) {
  a = (a + i) % 31;
}
timer.stop(id);

std::cout << timer << std::endl;
std::cout << "The tick is " << timer.tick() * 1000.0 << " ms" << std::endl;
```
which generates the output
```shell-session
Timer: (Computation=0.00309944 ms, Send=0 ms, Receive=0 ms, Idle=0 ms, Total=0.00309944 ms)
The tick is 0.001 ms
```
If `start` and `stop` are not called in matching pairs the behavior is undefined.


---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.


