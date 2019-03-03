#ifndef CULAYER_HPP
#define CULAYER_HPP

#include "particle.hpp"
#include <cuda.h>

__global__ void particle_step_kernel(int n, Particle *particles,
                                     float const *const sigs_in,
                                     float const *const absorption_rates_in,
                                     float *const weights_absorbed_in);
#endif
