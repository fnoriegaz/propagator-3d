#include "kernels.cuh"


__global__ void kernel_add_source(PropagatorSetup *setup, integer_t src_idx, int t){

	integer_t src_x = setup->sr_geometry->src_positions[3*src_idx];
	integer_t src_y = setup->sr_geometry->src_positions[3*src_idx+1]*setup->propagator_config->nx;
	integer_t src_z = setup->sr_geometry->src_positions[3*src_idx+2]*setup->propagator_config->nx*setup->propagator_config->ny;

	setup->propagator_fields->p[src_x+src_y+src_z] += setup->sr_geometry->d_source[t];

}

__global__ void kernel_cpml(PropagatorSetup *setup){

	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;

	real_t cpml_width = setup->propagator_config->cpml_width;
	real_t lx = cpml_width * setup->propagator_config->delta_x;
	real_t ly = cpml_width * setup->propagator_config->delta_y;
	real_t lz = cpml_width * setup->propagator_config->delta_z;

	real_t d0_x = -3 * log(setup->propagator_config->r) / (2 * lx);
	real_t d0_y = -3 * log(setup->propagator_config->r) / (2 * ly);
	real_t d0_z = -3 * log(setup->propagator_config->r) / (2 * lz);

	real_t fx = 0.0;
	real_t fy = 0.0;
	real_t fz = 0.0;

	real_t alpha_x = 0.0;
	real_t alpha_y = 0.0;
	real_t alpha_z = 0.0;

	real_t dx,dy,dz;

	//compute x axis cpml
	if(tx < cpml_width){
		fx = setup->propagator_config->delta_x * (cpml_width-tx-1);
	}
	dx = d0_x * setup->propagator_config->max_vel * (fx/lx) * (fx/lx);
	alpha_x = PI * setup->propagator_config->freq * (lx-fx) / lx;

	setup->cpml->a_x[tx] = expf(-(dx + alpha_x) * setup->propagator_config->delta_t);
	setup->cpml->b_x[tx] = (dx / (dx + alpha_x)) * (setup->cpml->a_x[tx] - 1.0);

}

__global__ void kernel_dpdt(PropagatorSetup *setup){

	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;
	integer_t ty = threadIdx.y + blockIdx.y * blockDim.y;
	integer_t tz = threadIdx.z + blockIdx.z * blockDim.z;
	integer_t tid_global = tx + (ty + tz * setup->propagator_config->ny) * setup->propagator_config->nx;

	real_t r_velocity = setup->inversion_params->d_velocity[tid_global];
	real_t r_psi_vel_x = setup->cpml->psi_vel_x[tid_global];
	real_t r_psi_vel_y = setup->cpml->psi_vel_y[tid_global];
	real_t r_psi_vel_z = setup->cpml->psi_vel_z[tid_global];
	
	if(tx > 3 && tx < (setup->propagator_config->nx-3) && ty > 3 && ty < (setup->propagator_config->ny-3) && tz > 3 && tz < (setup->propagator_config->nz-3)){
		real_t dvx_dx = compute_stencil_forward(setup->propagator_fields->vel_x, tid_global, 1, setup->propagator_config->delta_x);
		real_t dvy_dy = compute_stencil_forward(setup->propagator_fields->vel_y, tid_global, setup->propagator_config->nx, setup->propagator_config->delta_y);
		real_t dvz_dz = compute_stencil_forward(setup->propagator_fields->vel_z, tid_global, setup->propagator_config->nx*setup->propagator_config->ny, setup->propagator_config->delta_z);

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
	
	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;
	integer_t ty = threadIdx.y + blockIdx.y * blockDim.y;
	integer_t tz = threadIdx.z + blockIdx.z * blockDim.z;
	integer_t tid_global = tx + (ty + tz * setup->propagator_config->ny) * setup->propagator_config->nx;

	real_t r_psi_x = setup->cpml->psi_x[tid_global];
	real_t r_psi_y = setup->cpml->psi_y[tid_global];
	real_t r_psi_z = setup->cpml->psi_z[tid_global];
	

	if(tx > 2 && tx < (setup->propagator_config->nx-4) && ty > 2 && ty < (setup->propagator_config->ny-4) && tz > 2 && tz < (setup->propagator_config->nz-4)){

		real_t dp_dx = compute_stencil_backward(setup->propagator_fields->p, tid_global, 1, setup->propagator_config->delta_x);
		real_t dp_dy = compute_stencil_backward(setup->propagator_fields->p, tid_global, setup->propagator_config->nx, setup->propagator_config->delta_y);
		real_t dp_dz = compute_stencil_backward(setup->propagator_fields->p, tid_global, setup->propagator_config->nx*setup->propagator_config->ny, setup->propagator_config->delta_z);

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

