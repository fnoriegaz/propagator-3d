#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda.h>

#include "fdtd.h"
#include "propagator_structs.h"
#include "../include/propagator/propagator_constants.h"


__global__ void kernel_dpdt(PropagatorSetup *setup);

#endif
