#ifndef CULAYER_HPP
#define CULAYER_HPP

#include "particle.hpp"
#include <cuda.h>

__global__ void particle_step_kernel(int n,
                                     Particle *particles,
                                     int steps,
                                     float const *const sigs_in,
                                     float const *const absorption_rates_in,
                                     float *const weights_absorbed_out,
                                     int min_index,
                                     int max_index,
                                     float dx);
#endif
