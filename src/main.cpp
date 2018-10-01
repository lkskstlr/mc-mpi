#include "mcmpi_options.hpp"
#include "timer.hpp"
#include "worker.hpp"
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

void parse_input(int argc, char **argv, int world_rank, size_t *nb_particles) {
  if (argc != 2 && world_rank == 0) {
    fprintf(stderr, "Usage: mpirun -n nb_cells %s nb_particles\n", argv[0]);
    exit(1);
  }

  *nb_particles = atoi(argv[1]);

  if (*nb_particles <= 0 && world_rank == 0) {
    fprintf(stderr, "Wrong number of particles: %zu\n", *nb_particles);
    exit(1);
  }
}

int main(int argc, char **argv) {
  // constants
  MCMPIOptions opt;
  opt.x_min = 0.0;
  opt.x_max = 1.0;
  opt.x_ini = sqrt(2.) / 2;
  opt.buffer_size = pow(2, 24);
  opt.nb_cells_per_layer = 2;
  opt.cycle_time = 1e-5;
  opt.cycle_nb_steps = 10000;
  opt.statistics_cycle_time = opt.cycle_time * 10;

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  opt.world_size = world_size;

  // parse input
  parse_input(argc, argv, world_rank, &(opt.nb_particles));

  Worker worker(world_rank, opt);

  printf("Worker [%3d/%3d]: (%f, %f)\n", world_rank, world_size,
         worker.layer.x_min, worker.layer.x_max);
  double starttime = MPI_Wtime();
  std::vector<Timer::State> timer_states = worker.spin();
  double timespan = MPI_Wtime() - starttime;

  printf("Layer [%3d/%3d]: n_stats = %zu, absorbed weight = (\n", world_rank,
         world_size, timer_states.size());
  for (auto w : worker.weights_absorbed()) {
    printf("%f, ", w);
  }
  printf(")\n");

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * world_rank);

  std::cout << "p = " << world_rank << ": " << worker.timer
            << ", t = " << timespan * 1000.0 << " ms, " << std::endl;

  MPI_Finalize();
  return 0;
}
