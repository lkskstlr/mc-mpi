#include <cuda_runtime.h>
#include "gpu_errcheck.hpp"

int main(int argc, char ** argv)
{
    int* ptr;
    size_t petabyte = 1000000000000000;
    gpu_errcheck( cudaMalloc((void**)&ptr, petabyte) );
    return 0;
}
