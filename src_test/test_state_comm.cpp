#include <mpi.h>
#include <unistd.h>
#include <iostream>
#include <numeric>
#include "state_comm.hpp"

int main(int argc, char const *argv[]) {
  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int tag = 0;
  StateComm comm(world_size, world_rank, tag,
                 [world_size](std::vector<int> msgs) {
                   int sum = std::accumulate(msgs.begin(), msgs.end(), 0);
                   if (sum == world_size - 1) {
                     return StateComm::State::Finished;
                   }
                   return StateComm::State::Running;
                 });

  if (world_rank > 0) {
    comm.send_msg(1);
  }
  while (true) {
    usleep(100000);
    if (world_rank == 0) {
      comm.send_state();
    }

    if (comm.recv_state() == StateComm::State::Finished) {
      break;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    std::cout << "CORRECT" << std::endl;
  }
  MPI_Finalize();

  return 0;
}