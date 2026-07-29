#include "kernels.cuh"

	
__global__ void kernel_dpdt(Pressure *P, Vel *vel, float *density, float *velocity, GridConfig config, CPML *cpml){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	int tz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid_global = tx + (ty + tz * config.ny) * config.nx;

	float r_velocity = velocity[tid_global];
	float r_psi_vel_x = P->psi_vel_x[tid_global];
	float r_psi_vel_y = P->psi_vel_y[tid_global];
	float r_psi_vel_z = P->psi_vel_z[tid_global];
	
	if(tx > 4 && tx < (config.nx-3) && ty > 4 && ty < (config.ny-3) && tz > 4 && tz < (config.nz-3)){
		float dvx_dx = compute_stencil_forward(vel->x, tid_global, 1, config.delta_x);
		float dvy_dy = compute_stencil_forward(vel->y, tid_global, config.nx, config.delta_y);
		float dvz_dz = compute_stencil_forward(vel->z, tid_global, config.nx*config.ny, config.delta_z);

		r_psi_vel_x = r_psi_vel_x * cpml->a_x[tx] + b_x[x] * dvx_dx;
		r_psi_vel_y = r_psi_vel_y * cpml->a_y[ty] + b_y[y] * dvy_dy;
		r_psi_vel_z = r_psi_vel_z * cpml->a_z[tz] + b_z[z] * dvz_dz;

		p[tid_global] = p[tid_global] - config.delta_t * r_velocity * r_velocity * density[tid_global] * (
			dvx_dx + dvy_dy + dvz_dz;
		);
	}

}

void propagate(Pressure *P, Vel *vel, float *density, float *velocity, GridConfig config){

	dim3 blocks(8,8,8);
	dim3 grids((config.nx+blocks.x-1) / blocks.x,(config.ny+blocks.y-1) / blocks.x,(config.nz+blocks.z-1) / blocks.z);

	for(int t=0;t<config.time_samples;t++){
		kernel_dpdt<<<grids,blocks>>>(P,vel,density,velocity,config);
	}

}
