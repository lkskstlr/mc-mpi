#include "logging.h"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MCMPI_DEBUG_INIT(world_rank)
  MCMPI_DEBUG("My world rank is %d", world_rank)

  MCMPI_DEBUG_STOP()
  MPI_Finalize();
  return 0;
}