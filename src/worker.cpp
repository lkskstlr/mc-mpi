#include "worker.hpp"

Worker::Worker(int world_rank, const McOptions &options)
    : world_rank(world_rank), options(options), dist(SOME_SEED + world_rank),
      layer(decompose_domain(dist, options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_particles)),
      particle_comm(options.world_size, world_rank, options.buffer_size){};