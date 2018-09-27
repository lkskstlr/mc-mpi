CC=mpicc
CXX=mpicxx
CFLAGS=-I../include/ -O3

all:mpi_test logging logging2 logging3 timing

.PHONY: test
test: test_logging

.PHONY: test_logging
test_logging: logging2 logging3
	./test_logging

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
	rm -f mpi_test timing logging logging2 logging3 logs/*
