#include "layer.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <unistd.h>

using std::cout, std::endl;
using std::size_t;
using my_clock = std::chrono::high_resolution_clock;

#define MC_MPI_WAIT_MS 1

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
  const real_t x_min = 0.0;
  const real_t x_max = 1.0;
  const real_t x_ini = sqrt(2.) / 2;
  const size_t buffer_size = pow(2, 24);
  size_t nb_particles;
  parse_input(argc, argv, &nb_particles);

  // Generate Uniform (0,1) Distribution with standard seed
  UnifDist dist = UnifDist();

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  Layer layer = decompose_domain(dist, x_min, x_max, x_ini, world_size,
                                 world_rank, nb_particles);
  ParticleComm comm(world_size, world_rank, 12 * 1024 * 1024);
  std::vector<Particle> particles_left, particles_right;

  auto start = my_clock::now();
  auto finish = my_clock::now();

  while (true) {
    start = my_clock::now();
    // simulate
    layer.simulate(dist, 100'000'000, particles_left, particles_right);

    // send
    if (world_rank == 0) {

      comm.send_particles(particles_right, 1);
      particles_right.clear();
    } else if (world_rank == world_size - 1) {

      comm.send_particles(particles_left, -1);
      particles_left.clear();
    } else {

      comm.send_particles(particles_left, -1);
      particles_left.clear();
      comm.send_particles(particles_right, 1);
      particles_right.clear();
    }

    // receive
    comm.receive_particles(layer.particles);

    // timing
    finish = my_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;

    if (elapsed.count() < MC_MPI_WAIT_MS) {
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(MC_MPI_WAIT_MS) - elapsed);
    }
  }

  MPI_Finalize();

  return 0;
}
