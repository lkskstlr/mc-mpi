#include "layer.hpp"
#include <math.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {
  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int world_size = 1;
  constexpr int world_rank = 0;
  constexpr int nb_cells_per_layer = 100;
  constexpr int nb_particles = 100;
  constexpr real_t particle_min_weight = 0.0;

  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  std::vector<Particle> particles_left;
  std::vector<Particle> particles_right;
  std::vector<Particle> particles_disabled;
  layer.simulate(1'000'000'000, particles_left, particles_right,
                 particles_disabled);
  printf("layer.particles.size() = %zu\n", layer.particles.size());
  printf("particles_left.size() = %zu\n", particles_left.size());
  printf("particles_right.size() = %zu\n", particles_right.size());
  printf("particles_disabled.size() = %zu\n", particles_disabled.size());
  layer.dump_WA();
  return 0;
}