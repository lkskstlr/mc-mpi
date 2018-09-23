#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "mpi.h"
#include "types.hpp"

/* Is POD to be easily send over MPI */
typedef struct particle_tag {
public:
  real_t x;
  real_t mu;
  real_t wmc;
} Particle;
#endif