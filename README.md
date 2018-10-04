# mc-mpi

## Build & Run
Standard CMake project. This assumes you use a Unix system.
```shell-session
git clone https://github.com/lkskstlr/mc-mpi.git
cd mc-mpi
export CXX=mpicxx
mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make;
mpirun -n 5 ./main ../config.yaml
```
The third line *sets the CXX environment variable* to an mpi compatible compiler. The one but last line runs on 5 processes with parameters specified in `config.yaml`.

The results will be saved in `build/out` which should not be touched by the user. If you want to save this specific run, call `./save_run` from the build directory and the run will be saved to `../saved_runs/<timestamp>.tar.gz`. Also the `latest` symlink in `saved_runs` will be set for convenience.


## Reports
After a run you can generate a report by `cd tex` (from base directory) and calling `./report single_run ../saved_runs/latest` which will generate the report PDF in `tex/build/single_run.pdf`. The `report` script needs the python dependencies in `tex/requirements.txt`. There is a bug in/with matplotlib 3.0.0 so version 2.2.2 is recommended.

## Tests
A minimal testing script can be found in `test` and can be called by `./test`. A possible output is
```shell-session
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 test_async_comm
make -j4 test_serialization
make -j4 test_particle_step
Running tests. Should print CORRECT after at most x seconds:
    Test Async Comm    ( 1 sec): CORRECT
    Test Serialization ( 1 sec): CORRECT
    Test Particle Step ( 1 sec): CORRECT
COMPLETE
```
i.e. one indented line per test. If it says ERROR (or abort or an error message), the test will internally have failed and noticed the failure. If it prints nothing for more than x sec (here x = 1) there is probably a deadlock and the test didn't pass.

## Software Overview

### Classes
| Class         | Header | @brief      |
|---------------|--------|-------------|
| UnifDist      |random.hpp | Uniform (0,1] distribution based on mersenne twister. |
| Layer         |layer.hpp  | Representing one layer which has multiple cells in the simulation. Physics are implemented here. |
| Worker        |worker.hpp | Manages the control flow of one mpi process. Has members of type  `Layer`and `AsyncComm`. Each process instantiates exactly one worker. |
| AsyncComm&lt;T&gt; |async_comm.hpp| Encapsulation of asynchronous, buffered mpi communication. Internally handles all buffering. Templated. |

### MPI Strategy
Because there is no interaction among the particles, the processes are not synchronized in any way. When the process has completed a certain number of simulation steps, he will send the particles which left the layer to his neighbors by a buffered `MPI_Issend` and the `AsyncComm` will handle the buffer. Each process does the following steps
```
simulate -> send -> receive -> receive events -> simulate -> ...
```
The `send`, `receive` and `receive events` return immediately. The events are the communication between master and all other nodes, to determine if the simulation is over.

### Settings
All settings are passed via specifying a `config.yaml` and passing its filepath as the single argument to `main` as shown in Build & Run.


### Timing
A minimal timing class is provided in `include/timer.hpp` and can be used as follows:
```cpp
Timer timer;
const auto timestamp = timer.start(Timer::Tag::Computation);
int a = 1;
for (int i = 0; i < 1000; ++i) {
  a = (a + i) % 31;
}
timer.stop(timestamp);

std::cout << timer << std::endl;
std::cout << "The tick is " << timer.tick() * 1000.0 << " ms" << std::endl;
```
which generates the output
```shell-session
Timer: (Computation=0.00309944 ms, Send=0 ms, Receive=0 ms, Idle=0 ms, Total=0.00309944 ms)
The tick is 0.001 ms
```

The recommended way to utilize `Timer` for more complex scenarios is:
```cpp
Timer timer;

auto timestamp = timer.start(Timer::Tag::Computation);
// Computation
timer.change(timestamp, Timer::Tag::Send);
// Sending
timer.change(timestamp, Timer::Tag::Recv);
//Receiving

timer.stop(timestamp);

```
Because this way the sum of the computation time and send time will be equal to timing the both blocks together.

---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.