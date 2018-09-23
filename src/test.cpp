#include "layer.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include <chrono> // for high_resolution_clock
#include <iostream>
#include <mpi.h>
#include <unistd.h>

using std::cout, std::endl;

int main(int argc, char const *argv[]) {

  // Test Random
  UnifDist dist = UnifDist();
  // cout << dist() << ", " << dist() << ", " << dist() << endl;
  // cout << "sizeof(UnifDist) = " << sizeof(dist) << endl;

  // // Test Layer
  // Layer layer(0.0, 1.0);
  // cout << "sizeof(Particle) = " << sizeof(Particle) << endl;
  // cout << "Layer: n = "
  //      << ", x_min = " << layer.x_min << ", x_max = " << layer.x_max << endl;
  // // cout << dist() << ", " << dist() << ", " << dist() << endl;
  // layer.create_particles(dist, 0.5, 20);
  // cout << dist() << ", " << dist() << ", " << dist() << endl;

  // cout << "Particles:" << endl;
  // for (auto const &particle : layer.particles) {
  //   cout << "  x = " << particle.x << ", mu = " << particle.mu << endl;
  // }

  // -- Test ParticleComm --
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  ParticleComm comm(world_size, world_rank, 12 * 1024 * 1024);

  if (world_rank == 0) {
    Layer layer(0.0, 1.0);
    cout << "sizeof(Particle) = " << sizeof(Particle) << endl;
    cout << "Layer: n = "
         << ", x_min = " << layer.x_min << ", x_max = " << layer.x_max << endl;
    // cout << dist() << ", " << dist() << ", " << dist() << endl;
    layer.create_particles(dist, 0.5, 100'000);

    // cout << "Particles:" << endl;
    // for (auto const &particle : layer.particles) {
    //   cout << "  x = " << particle.x << ", mu = " << particle.mu
    //        << ", wmc = " << particle.wmc << endl;
    // }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      comm.send_particles(layer.particles, 1);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed sending time: " << elapsed.count() << " s\n";

  } else {
    sleep(2);
    bool flag = false;
    std::size_t n = 0;
    std::vector<Particle> parts;
    parts.reserve(1'000'000);
    auto start = std::chrono::high_resolution_clock::now();
    flag = comm.receive_particles(parts);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    cout << "received " << parts.size() << " particles" << endl;
    std::cout << "Elapsed receiving time: " << elapsed.count() << " s\n";
  }

  MPI_Finalize();
  return 0;
}
