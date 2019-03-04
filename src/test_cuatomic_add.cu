#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>



#define DIV_UP(x,y) (1 + ((x - 1) / y))

__global__ void sum_kernel(int n, float const*const ptr, float *const out) {
    extern __shared__ float sum[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0){
        sum[0] = 0;
    }
    __syncthreads();

    if(i < n){
        atomicAdd(sum, ptr[i]);
    }
    __syncthreads();

    if (threadIdx.x == 0){
        printf("Block %d, sum = %f\n", blockIdx.x, sum[0]);
    }
    __syncthreads();

    if (threadIdx.x == 0){
        atomicAdd(out, sum[0]);
    }
  
}
  
int main(int argc, char** argv) {
  int n = argc == 2 ? atoi(argv[1]) : 10000;

  float *const ptr = (float*) malloc(sizeof(float) * n);
  float* out = (float*) malloc(sizeof(float));
  *out = 0;
  float *d_ptr, *d_out;
  cudaMalloc((void**)&d_ptr, sizeof(float) * n);
  cudaMalloc((void**)&d_out, sizeof(float));

  for (int i = 0; i < n; i++){
      ptr[i] = 1.0 + (2.6458 / (float)((i+20)%1024+1));
  }
  float check = 0.0;
  for (int i = 0; i < n; i++) check += ptr[i];
  for (int i = 0; i < n; i++) ptr[i] /= check;

  cudaMemcpy(d_ptr, ptr, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, sizeof(float), cudaMemcpyHostToDevice);

  sum_kernel<<<DIV_UP(n, 1024), 1024, 100*sizeof(float)>>>(n, d_ptr, d_out);

  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);


check = 0.0;
  for (int i = 0; i < n; i++) check += ptr[i];
  printf("CPU : %f\n", check);
  printf("GPU : %f\n", *out);
  printf("DIFF: %.15f\n", check-*out);
  return 0;
}
