#include "async_comm.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>

using std::chrono::high_resolution_clock;

int main(int argc, char const *argv[]) {
  constexpr size_t num_ints = 10;

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AsyncComm<int> comm;
  size_t max_buffer_size =
      static_cast<size_t>(world_size) * num_ints * sizeof(int);
  comm.init(world_rank, MPI_INT, max_buffer_size);

  std::vector<int> data_send(num_ints, world_rank);
  std::vector<int> data_recv;

  if (world_rank > 0) {
    comm.send(data_send, world_rank - 1, 0);
  }

  while (true) {
    comm.recv(data_recv, MPI_ANY_SOURCE, MPI_ANY_TAG);
    if (data_recv.size() == (world_size - world_rank - 1) * num_ints) {
      // received all
      if (world_rank > 0) {
        comm.send(data_recv, world_rank - 1, 0);
      }
      break;
    }
  }

  MPI_Finalize();
  return 0;
}
