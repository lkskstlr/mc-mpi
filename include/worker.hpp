#ifndef WORKER_HPP
#define WORKER_HPP

#include "async_comm.hpp"
#include "layer.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include "types.hpp"

#define MCMPI_PARTICLE_TAG 1
#define MCMPI_NB_DISABLED_TAG 2
#define MCMPI_FINISHED_TAG 3
#define MC_MPI_WAIT_MS 1

typedef struct mc_options_tag {
  int world_size;
  real_t x_min, x_max, x_ini;

  std::size_t buffer_size;
  std::size_t nb_particles;
} McOptions;

class Worker {
public:
  Worker(int world_rank, const McOptions &McOptions);

  void spin();

  const int world_rank;
  const McOptions options;
  UnifDist dist;
  Layer layer;
  AsyncComm<Particle> particle_comm;
  std::vector<Particle> particles_left, particles_right;
  AsyncComm<int> event_comm;
};
#endif