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
  int nb_particles = 1000000;
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


  struct timeval  tv1, tv2;
  
  gettimeofday(&tv1, NULL);
  layer.simulate(-1);
  gettimeofday(&tv2, NULL);
  double time_cpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

  gettimeofday(&tv1, NULL);
  cusimulate(nb_particles, cu_particles, cu_sigs, cu_absorption_rates, cu_weights_absorbed, 0, nb_cells_per_layer, layer.dx);
  gettimeofday(&tv2, NULL);
  double time_gpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

  printf ("CPU = %f seconds\n", time_cpu);
  printf ("GPU = %f seconds\n", time_gpu);


  for (int j = 0; j < nb_cells_per_layer; j++){
    if (fabs(layer.weights_absorbed[j]-cu_weights_absorbed[j]) > FLOAT_CMP_PREC) {
      fprintf(stderr, "Weights absorbed at j = %d are not equal\n", j);
      fprintf(stderr, "CPU: %f\n", layer.weights_absorbed[j]);
      fprintf(stderr, "GPU: %f\n", cu_weights_absorbed[j]);
      exit(EXIT_FAILURE);
    }
  }

  return 0;
}
