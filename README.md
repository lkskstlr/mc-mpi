# mc-mpi

## Build & Run
Standard CMake project. This assumes you use a unix system.
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
i.e. one indented line per test. If it says FAILURE (or abort or an error message), the test will internally have failed and noticed the failure. If it prints nothing for more than x sec (here x = 1) there is probably a deadloc and the test didn't pass.

---
The project can be opened with sublime text. A build system and settings for EasyClangComplete are set.


