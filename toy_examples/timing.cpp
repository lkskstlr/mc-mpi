#include "timer.hpp"
#include <iostream>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#ifdef MPI_WTIME_IS_GLOBAL
  printf("MPI_WTIME_IS_GLOBAL\n");
#endif
  printf("MPI_Wtick = %f ms\n", MPI_Wtick() * 1000.0);

  /* Test MPI Timer */
  Timer timer;
  const auto timestamp = timer.start(Timer::Tag::Computation);
  int a = 1;
  for (int i = 0; i < 1000; ++i) {
    a = (a + i) % 31;
  }
  timer.stop(timestamp);

  std::cout << timer << std::endl;
  std::cout << "The tick is " << timer.tick() * 1000.0 << " ms" << std::endl;

  return 0;
}