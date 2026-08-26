#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda.h>

#include "fdtd.h"
#include "propagator_structs.h"
#include "../include/propagator/propagator_constants.h"
#include "kernels_constants.cuh"

__global__ void kernel_add_source(PropagatorSetup *setup, int src_idx, int t);

__global__ void kernel_cpml(real_t *a, real_t *b, integer_t n, real_t freq, real_t r, real_t delta_x, real_t delta_t, real_t max_vel, int cpml_width);

__global__ void kernel_dpdt(PropagatorSetup *setup);

__global__ void kernel_dvdxyz(PropagatorSetup *setup);

#endif
