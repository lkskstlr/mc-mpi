#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include "mcmpi_options.hpp"
#include "worker.hpp"

std::string parse_input(int argc, char **argv, int world_rank) {
  if (argc != 2 && world_rank == 0) {
    fprintf(stderr, "Usage: mpirun -n nb_layers %s config_file_path\n",
            argv[0]);
    exit(1);
  }

  std::string filepath(argv[1]);
  return filepath;
}

int main(int argc, char **argv) {
  // -- MPI Setup --
  MPI_Init(&argc, &argv);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // parse input
  std::string filepath = parse_input(argc, argv, world_rank);

  Worker worker = worker_from_config(filepath, world_size, world_rank);

  worker.spin();
  worker.dump();

  MPI_Finalize();
  return 0;
}
