#ifndef KERNELS_CU
#define KERNELS_CU

#include <cuda_runtime.h>
#include <cuda.h>

#include "fdtd.h"

__global__ void kernel_dpdt(float *p, Vel *vel, float *density, float *velocity, GridConfig config);

#endif
