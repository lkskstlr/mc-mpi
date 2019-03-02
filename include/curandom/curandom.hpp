#ifndef CURANDOM_HPP
#define CURANDOM_HPP

#include <cuda.h>
#include <types.hpp>

__global__ void rnd_real_kernel(int n, seed_t *seeds, float *reals);

#endif
