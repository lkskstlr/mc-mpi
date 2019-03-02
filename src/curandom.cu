#include "curandom.hpp"

__constant__ seed_t RNG_G = (seed_t)(6364136223846793005ull);
__constant__ seed_t RNG_C = (seed_t)(1442695040888963407ull);
__constant__ seed_t RNG_P = (seed_t)(1) << 63;

__global__ void rnd_real_kernel(int n, seed_t *seeds, float *reals) {
  float inv_RNG_P = (float)(1.0) / (float)((seed_t)(1) << 63);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    seeds[i] = (RNG_G * seeds[i] + RNG_C) % RNG_P;
    reals[i] = seeds[i] * inv_RNG_P;
  }
}
