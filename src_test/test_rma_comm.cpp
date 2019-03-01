#define _BSD_SOURCE

#include "rma_comm.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <unistd.h>

int main(int argc, char **argv) {
  using std::cout;
  using std::endl;

  int world_rank, world_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    if (argc != 2) {
      fprintf(stderr,
              "Must be called like: %s nb_int.\nWhere nb_int is the number of "
              "integers to be transfered.\n",
              argv[0]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (world_size != 2) {
      fprintf(stderr, "Must be called with n=2 MPI processes.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  int nb_int = atoi(argv[1]);

  RmaComm<int> comm;
  comm.init(-1); // use max buffer
  comm.mpi_data_t = MPI_INT;

  if (world_rank == 0)
    comm.subscribe(1);
  else
    comm.advertise(0);

  if (world_rank == 0)
    comm.print();

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    RmaComm<int>::BufferInfo buffer = comm.recv_buffer_infos[1];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 1) {
    std::vector<int> data(nb_int, 17);
    comm.send(data, 0);
    comm.send(data, 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0)
    comm.print();

  if (world_rank == 0) {
    std::vector<int> data;
    comm.recv(data, 1);
    comm.print();
  }

  MPI_Finalize();
  return 0;
}