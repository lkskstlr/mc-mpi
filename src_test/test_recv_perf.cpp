#include "layer.hpp"
#include "particle_comm.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>

int main(int argc, char const *argv[]) {
  using std::cout, std::endl;
  using std::chrono::high_resolution_clock;

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  constexpr double cycle_time = 1e-3;
  constexpr int num_cycles = 10'000;
  int buffer_size = 100'000'000;
  int nb_particles = 1'000;
  int nb_reps = 1'000;

  std::vector<Particle> particles;
  Particle p{5127801, 0.17, 123.12, -1231.1, 21};
  if (world_rank == 0) {
    for (int i = 0; i < nb_particles; ++i) {
      particles.push_back(p);
    }
  }

  ParticleComm particle_comm(world_rank, buffer_size);

  MPI_Barrier(MPI_COMM_WORLD);
  // Send
  if (world_rank == 0) {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nb_reps; ++i) {
      particle_comm.send(particles, 1, 0);
    }
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    cout << "Send: " << elapsed.count() << " ms for " << nb_particles * nb_reps
         << " particles " << nb_reps << " reps" << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Recv
  if (world_rank == 1) {
    auto start = high_resolution_clock::now();
    particle_comm.recv(particles, MPI_ANY_SOURCE, 0);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    cout << "Recv: " << elapsed.count() << " ms for " << nb_particles * nb_reps
         << " particles in " << nb_reps << " reps" << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    cout << "rank = " << world_rank << ", size = " << particles.size() << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 1) {
    cout << "rank = " << world_rank << ", size = " << particles.size() << endl;
  }
  MPI_Finalize();
  return 0;
}