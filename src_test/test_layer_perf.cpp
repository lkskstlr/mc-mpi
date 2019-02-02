#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "layer.hpp"

int main(int argc, char const *argv[])
{
  using std::chrono::high_resolution_clock;

  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int world_size = 1;
  constexpr int world_rank = 0;
  constexpr int nb_cells_per_layer = 1000;
  constexpr int nb_particles = 1000000;
  constexpr real_t particle_min_weight = 0.0;

  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  std::vector<Particle> particles_left;
  std::vector<Particle> particles_right;
  std::vector<Particle> particles_disabled;

  auto start = high_resolution_clock::now();
  layer.simulate(1000000000, particles_left, particles_right,
                 particles_disabled);
  auto finish = high_resolution_clock::now();

  layer.dump_WA();

  // Compare
  //
  printf("%zu, %zu, %zu\n%zu\n", particles_left.size(), particles_right.size(),
         particles_disabled.size(), layer.particles.size());

  std::chrono::duration<double, std::milli> elapsed = finish - start;
  printf("Time = %f s\n", elapsed.count() / 1e3);
  return 0;
}