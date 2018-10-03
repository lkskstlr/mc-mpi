#include "mcmpi_options.hpp"
#include "timer.hpp"
#include "worker.hpp"
#include "yaml_loader.hpp"
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

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
  MPI_Init(NULL, NULL);
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
