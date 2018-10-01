#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "types.hpp"

/* Is POD to be easily send over MPI */
typedef struct particle_tag {
public:
  real_t x;
  real_t mu;
  real_t wmc;
  int index; /** Cell index of the particle. This must be inside the data
                structure. If x \approx y, where y is the boundary between two
                cells, it is hard to tell in which cell the particle is based on
                floating point inaccuracies. */
} Particle;

#endif