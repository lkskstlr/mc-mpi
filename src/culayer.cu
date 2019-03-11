#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "particle.hpp"
#include "culayer_kernel.hpp"
#include "culayer.hpp"
#include "gpu_errcheck.hpp"
#include <sys/time.h>
#include <math.h>

// https://devblogs.nvidia.com/using-shared-memory-cuda-cc/

#define DIV_UP(x,y) (1 + ((x - 1) / y))
#define FLOAT_CMP_PREC (1e-4)

void particle_sort(
  int n_in, 
  Particle* particles_in, 
  int* n_active, 
  Particle* particles_active,
  int* n_inactive,
  Particle* particles_inactive,
  int min_index,
  int max_index){

  for(int i = 0; i < n_in; i++)
  {
    if(particles_in[i].index < min_index || particles_in[i].index >= max_index)
    {
      particles_inactive[*n_inactive] = particles_in[i];
      (*n_inactive)++;
    }
    else
    {
      particles_active[*n_active] = particles_in[i];
      (*n_active)++;
    }
  }

  // for (int i = 0; i < *n_active; i++){
  //   if (particles_active[i].index < min_index || particles_active[i].index >= max_index){
  //     printf("Error at i = %d, index = %d active particle is wrong\n", i, particles_active[i].index);
  //     exit(1);
  //   }
  // }

  // for (int i = 0; i < *n_inactive; i++){
  //   if (particles_inactive[i].index >= min_index && particles_inactive[i].index < max_index){
  //     printf("Error at i = %d, index = %d inactive particle is wrong\n", i, particles_inactive[i].index);
  //     exit(1);
  //   }
  // }
}
void cusimulate(int n,
  Particle* particles,
  float const* const sigs,
  float const* const absorption_rates,
  float * const weights_absorbed,
  int min_index,
  int max_index,
  float dx)
  {
    int n_cells = max_index - min_index;
    
    int n_active = n;
    int n_inactive = 0;

    int steps = 325;

    Particle* particles_inactive = (Particle*) malloc(sizeof(Particle) * n);
    Particle* buffer = (Particle*) malloc(sizeof(Particle) * n);
    Particle* const buffer_original = buffer;
    Particle* const particles_original = particles;

    Particle* d_particles;
    gpu_errcheck( cudaMalloc((void**)&d_particles, sizeof(Particle) * n) );

    float *d_sigs, *d_absorption_rates, *d_weights_absorbed;
    gpu_errcheck( cudaMalloc((void**)&d_sigs, sizeof(float) * n_cells) );
    gpu_errcheck( cudaMalloc((void**)&d_absorption_rates, sizeof(float) * n_cells) );
    gpu_errcheck( cudaMalloc((void**)&d_weights_absorbed, sizeof(float) * n_cells) );

    gpu_errcheck( cudaMemcpy(d_sigs, sigs, sizeof(float) * n_cells, cudaMemcpyHostToDevice) );
    gpu_errcheck( cudaMemcpy(d_absorption_rates, absorption_rates, sizeof(float) * n_cells, cudaMemcpyHostToDevice) );
    gpu_errcheck( cudaMemcpy(d_weights_absorbed, weights_absorbed, sizeof(float) * n_cells, cudaMemcpyHostToDevice) );

    while(n_active > 0){
      // printf("active = %d, inactive = %d\n", n_active, n_inactive);
      gpu_errcheck( cudaMemcpy(d_particles, particles, sizeof(Particle) * n_active, cudaMemcpyHostToDevice) );

      particle_step_kernel<<<DIV_UP(n_active, 256), 256, sizeof(float) * 3 * n_cells >>>(
        n_active, d_particles, steps, d_sigs, d_absorption_rates, d_weights_absorbed, min_index, max_index, dx );
      gpu_errcheck( cudaPeekAtLastError() );

      gpu_errcheck( cudaMemcpy(particles, d_particles, sizeof(Particle) * n_active, cudaMemcpyDeviceToHost) );
      {
        int n_active_new = 0;
        particle_sort(n_active, particles, &n_active_new, buffer, &n_inactive, particles_inactive, min_index, max_index);
        n_active = n_active_new;
        Particle* tmp_ptr = particles;
        particles = buffer;
        buffer = tmp_ptr;
      }

      // for (int i = 0; i < n_inactive; i++){
      //   printf("%d\n", particles_inactive[i].index);
      // }

      // for (int i = 0; i < n_active; i++){
      //   if (particles[i].index < min_index || particles[i].index >= max_index){
      //     printf("Error at i = %d, index = %d particles\n", i, particles[i].index);
      //     exit(1);
      //   }
      // }
    
      // for (int i = 0; i < n_inactive; i++){
      //   if (particles_inactive[i].index >= min_index && particles_inactive[i].index < max_index){
      //     printf("Error at i = %d, index = %d particles_inactive\n", i, particles_inactive[i].index);
      //     exit(1);
      //   }
      // }
    }
    // printf("%d\n", n_active);
    
    // printf("\n");
    gpu_errcheck( cudaMemcpy(weights_absorbed, d_weights_absorbed, sizeof(float) * n_cells, cudaMemcpyDeviceToHost) );

    if (n_inactive != n){
      fprintf(stderr, "Should have inactivated all particles\n");
    }
    for (int i = 0; i < n_inactive; i++){
      particles_original[i] = particles_inactive[i];
    }

    cudaFree(d_sigs);
    cudaFree(d_weights_absorbed);
    cudaFree(d_absorption_rates);

    free(particles_inactive);
    free(buffer_original);
    return;
  }