#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "curandom.hpp"
#include "random.hpp"
#include "types.hpp"
#include <sys/time.h>

#define DIV_UP(x,y) (1 + ((x - 1) / y))

int main(int argc, char** argv) {
  int n = 100000000;
  printf("n = %d\n", n);


  float* reals = (float*)malloc(sizeof(float) * n);
  seed_t* seeds = (seed_t*)malloc(sizeof(seed_t) * n);

  float* reals_cu = (float*)malloc(sizeof(float) * n);
  seed_t* seeds_cu = (seed_t*)malloc(sizeof(seed_t) * n);

  seed_t seed = 30061994;
  for (int i = 0; i < n; i++){
   seeds[i] = rnd_seed(&seed);
   seeds_cu[i] = seeds[i];
  }

  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < n; i++){
    reals[i] = rnd_real(seeds+i);
  }
  gettimeofday(&tv2, NULL);
  double time_cpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

  /* Call CUDA kernel */
  float* d_reals_cu;
  seed_t* d_seeds_cu;

  size_t free_mem, total_mem;
  size_t B_to_MB = 1024*1024;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("Before Allocation: Free = %4zu MB, Total = %zu MB\n", free_mem/B_to_MB, total_mem/B_to_MB);
  cudaMalloc((void**)&d_reals_cu, sizeof(float) * n);
  cudaMalloc((void**)&d_seeds_cu, sizeof(seed_t) * n);
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("After  Allocation: Free = %4zu MB, Total = %zu MB\n", free_mem/B_to_MB, total_mem/B_to_MB);

  gettimeofday(&tv1, NULL);
  cudaMemcpy(d_reals_cu, reals_cu, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_seeds_cu, seeds_cu, sizeof(seed_t) * n, cudaMemcpyHostToDevice);

  rnd_real_kernel<<<DIV_UP(n, 1024), 1024>>>(n, d_seeds_cu, d_reals_cu);

  cudaMemcpy(reals_cu, d_reals_cu, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(seeds_cu, d_seeds_cu, sizeof(seed_t) * n, cudaMemcpyDeviceToHost);
  gettimeofday(&tv2, NULL);
  double time_gpu = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

  printf ("CPU = %f seconds\n", time_cpu);
  printf ("GPU = %f seconds\n", time_gpu);


  cudaFree(d_reals_cu);
  cudaFree(d_seeds_cu);

  /* Compare */
  for (int i = 0; i < n; i++){
    if (seeds[i] != seeds_cu[i]){
      fprintf(stderr, "Seeds not equal at i = %d, seeds[i] = %llu, seeds_cu[i] = %llu\n", i, seeds[i], seeds_cu[i]);
      exit(EXIT_FAILURE);
    }

    if (reals[i] != reals_cu[i]){
      fprintf(stderr, "Seeds not equal at i = %d, reals[i] = %f, reals_cu[i] = %f\n", i, reals[i], reals_cu[i]);
      exit(EXIT_FAILURE);
    }
  }

  return 0;
}
