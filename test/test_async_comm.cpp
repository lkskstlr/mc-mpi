#include "async_comm.hpp"
#include <chrono>
#include <mpi.h>
#include <stdio.h>
#include <thread>

using std::chrono::high_resolution_clock;

int main(int argc, char const *argv[]) {
  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AsyncComm<int> comm;
  comm.init(world_rank, MPI_INT, 1024);

  std::vector<int> data_send(10, world_rank);
  std::vector<int> data_recv;

  for (int i = 0; i < 5; ++i) {
    if (world_rank > 0) {
      comm.send(data_send, world_rank - 1, 0);
    }
  }

  while (true) {
    if (world_rank > 0) {
      if (comm.recv(data_recv, MPI_ANY_SOURCE, MPI_ANY_TAG)) {
        comm.send(data_recv, world_rank - 1, 0);
        data_recv.clear();
      }
    } else {
      // master
      comm.recv(data_recv, MPI_ANY_SOURCE, MPI_ANY_TAG);

      if (data_recv.size() == (world_size - 1) * 50) {
#ifndef NDEBUG
        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(110));
#endif
        printf("CORRECT\n");
        exit(0);
      }
    }

    // std::this_thread::sleep_for(std::chrono::duration<double,
    // std::milli>(100));
  }

  return 0;
}