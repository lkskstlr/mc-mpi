#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "particle.hpp"
#include "layer.hpp"
#include "culayer.hpp"
#include "gpu_errcheck.hpp"
#include <sys/time.h>
#include <math.h>

// https://devblogs.nvidia.com/using-shared-memory-cuda-cc/

#define DIV_UP(x,y) (1 + ((x - 1) / y))
#define FLOAT_CMP_PREC (1e-4)

inline bool particle_cmp(Particle x, Particle y){
  return
    (x.seed == y.seed) &&
    (x.index == y.index) &&
    (fabs(x.x - y.x) < FLOAT_CMP_PREC) &&
    (fabs(x.mu - y.mu) < FLOAT_CMP_PREC) &&
    (fabs(x.wmc - y.wmc) < FLOAT_CMP_PREC);
}

int main(int argc, char** argv) {
  int nb_particles = 100000;
  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int world_size = 1;
  constexpr int world_rank = 0;
  constexpr int nb_cells_per_layer = 1000;
  constexpr real_t particle_min_weight = 0.0;

  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));


  float* cu_sigs = (float*) malloc(sizeof(float)*nb_cells_per_layer);
  float* cu_absorption_rates = (float*) malloc(sizeof(float)*nb_cells_per_layer);
  float* cu_weights_absorbed = (float*) malloc(sizeof(float)*nb_cells_per_layer);
  Particle* cu_particles = (Particle*) malloc(sizeof(Particle)*nb_particles);

  memcpy(cu_sigs, layer.sigs.data(), sizeof(float)*nb_cells_per_layer);
  memcpy(cu_absorption_rates, layer.absorption_rates.data(), sizeof(float)*nb_cells_per_layer);
  memcpy(cu_weights_absorbed, layer.weights_absorbed.data(), sizeof(float)*nb_cells_per_layer);
  memcpy(cu_particles, layer.particles.data(), sizeof(Particle)*nb_particles);

  float total_cpu = 0.0;
  float total_gpu = 0.0;
  for (int i = 0; i < nb_particles; i++) total_cpu += layer.particles[i].wmc;
  for (int i = 0; i < nb_particles; i++) total_gpu += cu_particles[i].wmc;

  printf("CPU: %f\n", total_cpu);
  printf("GPU: %f\n", total_gpu);

  int steps = 10000;
  printf("steps = %d\n", steps);
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
  /* --- Simulate with Layer --- */
  for (int step = 0; step < steps; step++){
    for (int i = 0; i < nb_particles; i++){
      if (layer.particles[i].index >= 0 && layer.particles[i].index < nb_cells_per_layer){
        layer.particle_step(layer.particles[i], layer.weights_absorbed);
      }
    }
  }
  gettimeofday(&tv2, NULL);
  double time_cpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  

  /* --- Alloc on GPU --- */
  float *d_cu_sigs, *d_cu_absorption_rates, *d_cu_weights_absorbed;
  Particle* d_cu_particles;

  size_t free_mem, total_mem;
  size_t B_to_MB = 1024*1024;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("Before Allocation: Free = %4zu MB, Total = %zu MB\n", free_mem/B_to_MB, total_mem/B_to_MB);
  gpu_errcheck( cudaMalloc((void**)&d_cu_sigs, sizeof(float) * nb_cells_per_layer) );
  gpu_errcheck( cudaMalloc((void**)&d_cu_absorption_rates, sizeof(float) * nb_cells_per_layer) );
  gpu_errcheck( cudaMalloc((void**)&d_cu_weights_absorbed, sizeof(float) * nb_cells_per_layer) );
  gpu_errcheck( cudaMalloc((void**)&d_cu_particles, sizeof(Particle) * nb_particles) );
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("After  Allocation: Free = %4zu MB, Total = %zu MB\n", free_mem/B_to_MB, total_mem/B_to_MB);

  /* Copy to GPU */
  gettimeofday(&tv1, NULL);
  gpu_errcheck( cudaMemcpy(d_cu_sigs, cu_sigs, sizeof(float) * nb_cells_per_layer, cudaMemcpyHostToDevice) );
  gpu_errcheck( cudaMemcpy(d_cu_absorption_rates, cu_absorption_rates, sizeof(float) * nb_cells_per_layer, cudaMemcpyHostToDevice) );
  gpu_errcheck( cudaMemcpy(d_cu_weights_absorbed, cu_weights_absorbed, sizeof(float) * nb_cells_per_layer, cudaMemcpyHostToDevice) );
  gpu_errcheck( cudaMemcpy(d_cu_particles, cu_particles, sizeof(Particle) * nb_particles, cudaMemcpyHostToDevice) );



  /* Invoke Kernel */
  size_t n_shared_mem = sizeof(float)*3*nb_cells_per_layer;
  particle_step_kernel<<<DIV_UP(nb_particles, 1024), 1024, n_shared_mem>>>(
    nb_particles, d_cu_particles, steps, d_cu_sigs, d_cu_absorption_rates, d_cu_weights_absorbed);
  gpu_errcheck( cudaPeekAtLastError() );
  gpu_errcheck( cudaDeviceSynchronize() );


  /* Copy back from GPU */
  gpu_errcheck( cudaMemcpy(cu_weights_absorbed, d_cu_weights_absorbed, sizeof(float) * nb_cells_per_layer, cudaMemcpyDeviceToHost) );
  gpu_errcheck( cudaMemcpy(cu_particles, d_cu_particles, sizeof(Particle) * nb_particles, cudaMemcpyDeviceToHost) );
  gpu_errcheck( cudaDeviceSynchronize() );

  gettimeofday(&tv2, NULL);
  double time_gpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

  printf ("CPU = %f seconds\n", time_cpu);
  printf ("GPU = %f seconds\n", time_gpu);


  /* Compare */
  for (int i = 0; i < nb_particles; i++){
    if (!particle_cmp(layer.particles[i], cu_particles[i])){
      fprintf(stderr, "Particles at i = %d are not equal\n", i);
      fprintf(stderr, "%llu, %d, %f, %f\n", layer.particles[i].seed, layer.particles[i].index, layer.particles[i].x, layer.particles[i].mu);
      fprintf(stderr, "%llu, %d, %f, %f\n", cu_particles[i].seed, cu_particles[i].index, cu_particles[i].x, cu_particles[i].mu);
      exit(EXIT_FAILURE);
    }
  }

  for (int j = 0; j < nb_cells_per_layer; j++){
    if (fabs(layer.weights_absorbed[j]-cu_weights_absorbed[j]) > FLOAT_CMP_PREC) {
      fprintf(stderr, "Weights absorbed at j = %d are not equal\n", j);
      fprintf(stderr, "CPU: %f\n", layer.weights_absorbed[j]);
      fprintf(stderr, "GPU: %f\n", cu_weights_absorbed[j]);
      exit(EXIT_FAILURE);
    }
  }

  total_cpu = 0.0;
  total_gpu = 0.0;
  for (int i = 0; i < nb_particles; i++) total_cpu += layer.particles[i].wmc;
  for (int i = 0; i < nb_particles; i++) total_gpu += cu_particles[i].wmc;

  printf("CPU: %f\n", total_cpu);
  printf("GPU: %f\n", total_gpu);

  int active_count_cpu = 0;
  int active_count_gpu = 0;
  for (int i = 0; i < nb_particles; i++)
    active_count_cpu += (int)(layer.particles[i].index >= 0 && layer.particles[i].index < nb_cells_per_layer);
  for (int i = 0; i < nb_particles; i++)
    active_count_gpu += (int)(cu_particles[i].index >= 0 && cu_particles[i].index < nb_cells_per_layer);

  printf("CPU: %d\n", active_count_cpu);
  printf("GPU: %d\n", active_count_gpu);
  return 0;
}
