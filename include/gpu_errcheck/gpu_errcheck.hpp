#ifndef GPU_ERRCHECK_HPP
#define GPU_ERRCHECK_HPP
// See:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define gpu_errcheck(ans)                                                      \
  { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#endif
