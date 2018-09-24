#ifndef WORKER_HPP
#define WORKER_HPP

#include "layer.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include "types.hpp"

typedef struct mc_options_tag {
  int world_size;
  real_t x_min, x_max, x_ini;

  std::size_t buffer_size;
  std::size_t nb_particles;
} McOptions;

class Worker {
public:
  Worker(int world_rank, const McOptions &McOptions);

  const int world_rank;
  const McOptions options;
  UnifDist dist;
  Layer layer;
  ParticleComm particle_comm;
  std::vector<Particle> particles_left, particles_right;
};
#endif