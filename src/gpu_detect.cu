#include "gpu_detect.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

int cuda_get_num_gpus(){
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("cuda_get_num_gpus cu version count = %d\n", count);

    return count;
}