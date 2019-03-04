#include "culayer.hpp"
#include <stdio.h>

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#define DIV_UP(x,y) (1 + ((x - 1) / y))


__constant__ seed_t RNG_G = (seed_t)(6364136223846793005ull);
__constant__ seed_t RNG_C = (seed_t)(1442695040888963407ull);
__constant__ seed_t RNG_P = (seed_t)(1) << 63;

__device__ __forceinline__ float cu_rnd_real(seed_t* seed) {
  float inv_RNG_P = (float)(1) / (float)(RNG_P);
  *seed = (RNG_G * *seed + RNG_C) % RNG_P;
  return (float)(*seed) * inv_RNG_P;
}

__global__ void particle_step_kernel(int n,
  Particle* particles,
  int steps,
  float const* const sigs_in,
  float const* const absorption_rates_in,
  float * const weights_absorbed_out,
  int min_index,
  int max_index,
  float dx)
{
  extern __shared__ float sdata[];

  int n_cells = max_index-min_index;

  float * const sigs = sdata;
  float * const absorption_rates = sdata + n_cells;
  float * const weights_absorbed = sdata + 2*n_cells;

  for (int j = 0; j < DIV_UP(n_cells, blockDim.x); j++){
    int cpy_ind = j*blockDim.x + threadIdx.x;
    if (cpy_ind < n_cells){
      sigs[cpy_ind] = sigs_in[cpy_ind];
      absorption_rates[cpy_ind] = absorption_rates_in[cpy_ind];
      weights_absorbed[cpy_ind] = 0;
    }
  }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n){
    Particle particle = particles[i];

    for (int step = 0; step < steps; step++){
      if (particle.index >= min_index && particle.index < max_index)
      {
        int local_index = particle.index - min_index;
        const float interaction_rate = 1.0 - absorption_rates[local_index];
        const float sig_a = sigs[local_index] * absorption_rates[local_index];
        const float sig_i = sigs[local_index] * interaction_rate;

        // calculate theoretic movement
        const float h = cu_rnd_real(&particle.seed);
        float di = MAXREAL;
        if (sig_i > EPS_PRECISION){
          // This should always be true
          di = -log(h) / sig_i;
        }

        // -- possible new cell --
        float mu_sign = copysignf(1.0, particle.mu);
        int index_new = __float2int_rn(mu_sign) + particle.index;
        float x_new_edge = particle.index * dx;
        if (mu_sign == 1){
          x_new_edge += dx;
        }

        float di_edge = MAXREAL;
        if (particle.mu < -EPS_PRECISION || EPS_PRECISION < particle.mu){
          di_edge = (x_new_edge - particle.x) / particle.mu;
        }

        if (di < di_edge) {
          /* move inside cell an draw new mu */
          index_new = particle.index;
          particle.x += di * particle.mu;
          particle.mu = 2 * cu_rnd_real(&particle.seed) - 1;
        } else {
          /* set position to border */
          di = di_edge;
          particle.x = x_new_edge;
        }

        // -- Calculate amount of absorbed energy --
        const float dw = (1 - expf(-sig_a * di)) * particle.wmc;

        /* Weight removed from particle is added to the layer */
        particle.wmc -= dw;
        atomicAdd(weights_absorbed + local_index, dw);
        particle.index = index_new;

        
      }
    }
    particles[i] = particle;
  }

  __syncthreads();
  for (int j = 0; j < DIV_UP(n_cells, blockDim.x); j++){
    int cpy_ind = j*blockDim.x + threadIdx.x;
    if (cpy_ind < n_cells){
      atomicAdd(weights_absorbed_out + cpy_ind, weights_absorbed[cpy_ind]);
    }
  }
}