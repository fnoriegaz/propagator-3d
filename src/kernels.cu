#include "kernels.cuh"


__global__ void kernel_add_source(PropagatorSetup *setup, int src_idx. int t){

	int src_x = setup->sr_geometry->src_positions[3*src_idx];
	int src_y = setup->sr_geometry->src_positions[3*src_idx+1]*setup->propagator_config->nx;
	int src_z = setup->sr_geometry->src_positions[3*src_idx+2]*setup->propagator_config->nx*setup->propagator_config->ny;

	setup->propagator_fields->p[src_x+src_y+src_z] += setup->sr_geometry->d_source[t];

}

__global__ void kernel_dpdt(PropagatorSetup *setup){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	int tz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid_global = tx + (ty + tz * setup->propagator_config->ny) * setup->propagator_config->nx;

	float r_velocity = setup->inversion_params->d_velocity[tid_global];
	float r_psi_vel_x = setup->cpml->psi_vel_x[tid_global];
	float r_psi_vel_y = setup->cpml->psi_vel_y[tid_global];
	float r_psi_vel_z = setup->cpml->psi_vel_z[tid_global];
	
	if(tx > 3 && tx < (setup->propagator_config->nx-3) && ty > 3 && ty < (setup->propagator_config->ny-3) && tz > 3 && tz < (setup->propagator_config->nz-3)){
		float dvx_dx = compute_stencil_forward(setup->propagator_fields->vel_x, tid_global, 1, setup->propagator_config->delta_x);
		float dvy_dy = compute_stencil_forward(setup->propagator_fields->vel_y, tid_global, setup->propagator_config->nx, setup->propagator_config->delta_y);
		float dvz_dz = compute_stencil_forward(setup->propagator_fields->vel_z, tid_global, setup->propagator_config->nx*setup->propagator_config->ny, setup->propagator_config->delta_z);

		//update psi fields rather than in separate kernel call
		r_psi_vel_x = r_psi_vel_x * setup->cpml->a_x[tx] + setup->cpml->b_x[tx] * dvx_dx;
		r_psi_vel_y = r_psi_vel_y * setup->cpml->a_y[ty] + setup->cpml->b_y[ty] * dvy_dy;
		r_psi_vel_z = r_psi_vel_z * setup->cpml->a_z[tz] + setup->cpml->b_z[tz] * dvz_dz;

		setup->propagator_fields->p[tid_global] = setup->propagator_fields->p[tid_global] -
			setup->propagator_config->delta_t * r_velocity * r_velocity * setup->inversion_params->d_density[tid_global] * (
				dvx_dx + dvy_dy + dvz_dz
		);

		setup->cpml->psi_vel_x[tid_global] = r_psi_vel_x;
		setup->cpml->psi_vel_y[tid_global] = r_psi_vel_y;
		setup->cpml->psi_vel_z[tid_global] = r_psi_vel_z;
	}

}

__global__ void kernel_dvdxyz(PropagatorSetup *setup){
	
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	int tz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid_global = tx + (ty + tz * setup->propagator_config->ny) * setup->propagator_config->nx;

	float r_psi_x = setup->cpml->psi_x[tid_global];
	float r_psi_y = setup->cpml->psi_y[tid_global];
	float r_psi_z = setup->cpml->psi_z[tid_global];
	

	if(tx > 2 && tx < (setup->propagator_config->nx-4) && ty > 2 && ty < (setup->propagator_config->ny-4) && tz > 2 && tz < (setup->propagator_config->nz-4)){

		float dp_dx = compute_stencil_backward(setup->propagator_fields->p, tid_global, 1, setup->propagator_config->delta_x);
		float dp_dy = compute_stencil_backward(setup->propagator_fields->p, tid_global, setup->propagator_config->nx, setup->propagator_config->delta_y);
		float dp_dz = compute_stencil_backward(setup->propagator_fields->p, tid_global, setup->propagator_config->nx*setup->propagator_config->ny, setup->propagator_config->delta_z);

		//update psi fields rather than in separate kernel call
		r_psi_x = r_psi_x * setup->cpml->a_x[tx] + setup->cpml->b_x[tx] * dp_dx;
		r_psi_y = r_psi_y * setup->cpml->a_y[ty] + setup->cpml->b_y[ty] * dp_dy;
		r_psi_z = r_psi_z * setup->cpml->a_z[tz] + setup->cpml->b_z[tz] * dp_dz;

		setup->propagator_fields->vel_x[tid_global] = setup->propagator_fields->vel_x[tid_global] -
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * dp_dx  + r_psi_x;

		setup->propagator_fields->vel_y[tid_global] = setup->propagator_fields->vel_y[tid_global] -
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * dp_dy  + r_psi_y;

		setup->propagator_fields->vel_z[tid_global] = setup->propagator_fields->vel_z[tid_global] -
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * dp_dz  + r_psi_z;

		setup->cpml->psi_x[tid_global] = r_psi_x;
		setup->cpml->psi_y[tid_global] = r_psi_y;
		setup->cpml->psi_z[tid_global] = r_psi_z;


	}

}

