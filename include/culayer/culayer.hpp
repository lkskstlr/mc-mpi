#ifndef CULAYER_HPP
#define CULAYER_HPP

#include "particle.hpp"

void cusimulate(int n,
                Particle *particles,
                float const *const sigs,
                float const *const absorption_rates,
                float *const weights_absorbed,
                int min_index,
                int max_index,
                float dx);

#endif
