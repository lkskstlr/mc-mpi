#include "layer.hpp"
#include "particle_comm.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

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
  int buffer_size = 1000'000'000;
  int nb_particles = 500'000;
  int nb_reps = 10;

  std::vector<Particle> particles;
  Particle p{5127801, 0.17, 123.12, -1231.1, 21};
  if (world_rank == 0) {
    for (int i = 0; i < nb_particles; ++i) {
      particles.push_back(p);
    }
  }

  double recv_times[5] = {0.0};
  int recv_packets = 0;

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
         << " particles " << nb_reps << " in reps" << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Recv
  if (world_rank == 1) {
    auto start = high_resolution_clock::now();
    particle_comm.recv_debug(particles, MPI_ANY_SOURCE, 0, recv_times,
                             &recv_packets);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    cout << "Recv: " << elapsed.count() << " ms for " << nb_particles * nb_reps
         << " particles in " << nb_reps << " reps" << endl;
    printf("recv_times = (%f, %f, %f, %f, %f) [secs]\n", recv_times[0],
           recv_times[1], recv_times[2], recv_times[3], recv_times[4]);
    printf("recv_times interpretation (0 = MPI_Probe, 1 = MPI_Get_Count, 2 = "
           "Buffer Handling, 3 = MPI_Recv, 4 = std::vec copy)\n");
  }
  if (world_rank == 0) {
    // master keep busy
    int k = 13;
    int s = 7;
    int upper = 0;
    if (argc > 1) {
      upper = atoi(argv[1]);
    }
    for (int j = 0; j < upper; ++j) {
      for (int i = 0; i < 1000'000'000; ++i) {
        s = ((s + (s % k)) << 1);
      }
    }
    // prevent the loop from being optimized out
    if (s == 1514198968) {
      printf(" \n");
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}