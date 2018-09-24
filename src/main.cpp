#include "worker.hpp"
#include <mpi.h>
#include <unistd.h>

void parse_input(int argc, char **argv, size_t *nb_particles) {
  if (argc != 2) {
    fprintf(stderr, "Usage: mpirun -n nb_cells %s nb_particles\n", argv[0]);
    exit(1);
  }

  *nb_particles = atoi(argv[1]);

  if (*nb_particles <= 0) {
    fprintf(stderr, "Wrong number of particles: %ld\n", *nb_particles);
    exit(1);
  }
}

int main(int argc, char **argv) {
  // constants
  McOptions opt;
  opt.x_min = 0.0;
  opt.x_max = 1.0;
  opt.x_ini = sqrt(2.) / 2;
  opt.buffer_size = pow(2, 24);
  parse_input(argc, argv, &(opt.nb_particles));

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  opt.world_size = world_size;

  Worker worker(world_rank, opt);
  printf("----- [%03d/%03d]: n = %08ld -----\n", world_rank, opt.world_size,
         worker.layer.particles.size());
  sleep(1);
  worker.spin();

  MPI_Finalize();
  return 0;
}
