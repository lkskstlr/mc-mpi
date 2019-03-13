#include "gpu_detect.hpp"
#include <cuda_runtime.h>

int cuda_get_num_gpus(){
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}