#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda.h>

#include "fdtd.h"
#include "propagator_structs.h"
#include "../include/propagator/propagator_constants.h"

__global__ void kernel_add_source(PropagatorSetup *setup, int src_idx, int t);

__global__ void kernel_dpdt(PropagatorSetup *setup);

__global__ void kernel_dvdxyz(PropagatorSetup *setup);

#endif
