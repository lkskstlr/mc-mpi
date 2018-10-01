CC=mpicc
CXX=mpicxx
CFLAGS=-I../include/ -O3

all:mpi_test logging logging2 logging3 timing assert_debug assert_release hash array_in_struct

.PHONY: test
test: test_logging

.PHONY: test_logging	
test_logging: logging2 logging3
	./test_logging

array_in_struct: array_in_struct.cpp
	$(CC) $(CFLAGS) -o $@ $^

hash: hash.cpp
	$(CXX) $(CFLAGS) -std=c++17 -o $@ $^

assert_debug: assert.cpp
	$(CXX) -g -o $@ $^

assert_release: assert.cpp
	$(CXX) -DNDEBUG -O3 -o $@ $^

mpi_test: mpi_test.c
	$(CC) $(CFLAGS) -o $@ $^

timing: timing.cpp ../src/timer.cpp
	$(CXX) $(CFLAGS) -std=c++17 -o $@ $^

logging: logging.c
	$(CC) $(CFLAGS) -o $@ $^

logging2: logging2.c
	gcc -I../include/ -D NDEBUG -o $@ $^

logging3: logging3.c
	gcc -o $@ $^

.PHONY: clean
clean:
	rm -f mpi_test timing logging logging2 logging3 logs/* assert_debug assert_release hash array_in_struct
	rm -rf assert_debug.dSYM
