#include "latched_comm.hpp"
#include <mpi.h>
#include <numeric>
#include <stdio.h>

int main(int argc, char const *argv[]) {
  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  LatchedComm comm(world_size, world_rank, 0);

  if (world_rank > 0) {
    // printf("%2d: Sent\n", world_rank);
    comm.send(1);
  }
  while (true) {
    if (world_rank == 0) {
      auto const &msgs = comm.msgs();
      if (std::accumulate(msgs.begin(), msgs.end(), 0) == world_size - 1) {
        // printf("%2d: Bcast\n", world_rank);
        comm.bcast(1);
        break;
      }
    } else {
      if (comm.msg() == 1) {
        // printf("%2d: Recv\n", world_rank);
        break;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    printf("CORRECT\n");
    exit(0);
  }
  MPI_Finalize();

  return 0;
}