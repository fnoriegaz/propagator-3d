#include "kernels.cuh"
#define STENCIL 4


__global__ void kernel_add_source(PropagatorSetup *setup, integer_t src_idx, int t){

	integer_t src_x = setup->sr_geometry->d_src_positions[3*src_idx];
	integer_t src_y = setup->sr_geometry->d_src_positions[3*src_idx+1]*setup->propagator_config->nx;
	integer_t src_z = setup->sr_geometry->d_src_positions[3*src_idx+2]*setup->propagator_config->nx*setup->propagator_config->ny;

	setup->propagator_fields->p[src_x+src_y+src_z] += setup->sr_geometry->d_source[t];

}


__global__ void kernel_cpml(real_t *a, real_t *b, integer_t n, real_t freq, real_t r, real_t delta_x, real_t delta_t, real_t max_vel, int cpml_width){

	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;

	real_t lx = cpml_width * delta_x;
	real_t d0_x = -3 * log(r) / (2 * lx);
	real_t fx = 0.0;
	real_t alpha_x = 0.0;
	real_t dx;

	if(tx < cpml_width){
		fx = delta_x * (cpml_width-tx-1);
	}

	if(tx > (n - cpml_width - 1) && tx < n){
		fx = (tx - (n - cpml_width)) * delta_x;
	}
	dx = d0_x * max_vel * (fx/lx) * (fx/lx);
	alpha_x = PI * freq * (lx-fx) / lx;

	a[tx] = expf(-(dx + alpha_x) * delta_t);
	b[tx] = (dx / (dx + alpha_x)) * (a[tx] - 1.0);

}

__global__ void kernel_dpdt(PropagatorSetup *setup){

	PropagatorConfig *__restrict__ config = setup->propagator_config;
	PropagatorFields *__restrict__ fields = setup->propagator_fields;
	InversionParams *__restrict__ params = setup->inversion_params;
	CPML *__restrict__ cpml = setup->cpml;

	integer_t s_tx = threadIdx.x;
	integer_t s_ty = threadIdx.y;
	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;
	integer_t ty = threadIdx.y + blockIdx.y * blockDim.y;
	integer_t tz;

	real_t front[4];
	real_t behind[4];


	//load behind and front once
	for(integer_t l=0;l<STENCIL;l++){
		integer_t tid_global_behind = tx + (ty + l * config->ny) * config->nx;
		integer_t tid_global_front = tx + (ty + (l+STENCIL) * config->ny) * config->nx;
		front[l] = fields->vel_z[tid_global_front];
		behind[STENCIL-1-l] = fields->vel_z[tid_global_behind];
	}
	for(tz=STENCIL;tz<config->nz;tz++){
		
		integer_t tid_global = tx + (ty + tz * config->ny) * config->nx;
		integer_t tid_global_front = tx + (ty + (tz+STENCIL) * config->ny) * config->nx;

		real_t r_velocity = params->d_velocity[tid_global];
		real_t r_psi_vel_x = cpml->psi_vel_x[tid_global];
		real_t r_psi_vel_y = cpml->psi_vel_y[tid_global];
		real_t r_psi_vel_z = cpml->psi_vel_z[tid_global];

		real_t dvx_dx = 0.;
		real_t dvy_dy = 0.;
		real_t dvz_dz = 0.;

		__shared__ real_t s_vel_x[(TPBX+8)*TPBY];
		__shared__ real_t s_vel_y[TPBX*(TPBY+8)];

		//copy center block inside shared memory
		if(tx < config->nx && ty < config->ny && tz < config->nz){
			s_vel_x[s_tx+STENCIL + s_ty*(TPBX+2*STENCIL)] = fields->vel_x[tid_global];
			s_vel_y[s_tx + (s_ty+STENCIL)*TPBX] = fields->vel_y[tid_global];
		}
		//copy left side halo inside shared memory. duplicating first STENCIL elements per x slice
		if(s_tx < STENCIL && tx >= STENCIL && ty < config->ny && tz < config->nz){
			s_vel_x[s_tx + s_ty*(TPBX+2*STENCIL)] = fields->vel_x[tid_global - STENCIL];
		}
		//copy right side halo inside shared memory. duplicating last STENCIL elements per x slice(+TPBX offset)
		if(s_tx < STENCIL && tx < (config->nx-TPBX) && ty < config->ny && tz < config->nz){
			s_vel_x[s_tx + s_ty*(TPBX+2*STENCIL)] = fields->vel_x[tid_global + TPBX];
		}
		//back side halo copy inside shared memory
		if(tx < config->nx && s_ty < STENCIL && ty >= STENCIL && tz < config->nz){
			s_vel_y[s_tx + s_ty*TPBX] = fields->vel_y[tid_global - STENCIL*config->nx];
		}
		//front side halo copy inside shared memory
		if(tx < config->nx && s_ty < STENCIL && ty < (config->ny-TPBY) && tz < config->nz){
			s_vel_y[s_tx + s_ty*TPBX] = fields->vel_y[tid_global + TPBY*config->nx];
		}
		__syncthreads();

		if(tx >= STENCIL && tx < (config->nx-STENCIL) && ty >= STENCIL && ty < (config->ny-STENCIL) && tz >= STENCIL && tz < (config->nz-STENCIL)){

			integer_t tid_shared_x = s_tx + STENCIL + s_ty*(2*STENCIL+TPBX);
			integer_t tid_shared_y = s_tx + (s_ty+STENCIL)*TPBX;

			dvx_dx = s_compute_stencil_forward(s_vel_x, tid_shared_x, 1, config->delta_x);
			dvy_dy = s_compute_stencil_forward(s_vel_y, tid_shared_y, TPBX, config->delta_y);

			//update psi fields rather than in separate kernel call
			r_psi_vel_x = r_psi_vel_x * cpml->a_x[tx] + cpml->b_x[tx] * dvx_dx;
			r_psi_vel_y = r_psi_vel_y * cpml->a_y[ty] + cpml->b_y[ty] * dvy_dy;
		}

		if(tx > 3 && tx < (config->nx-3) && ty > 3 && ty < (config->ny-3) && tz > 3 && tz < (config->nz-3)){

			//dvz_dz = compute_stencil_forward(fields->vel_z, tid_global, config->nx*config->ny, config->delta_z);
			dvz_dz = compute_stencil_forward_z(front, behind, config->delta_z);
			r_psi_vel_z = r_psi_vel_z * cpml->a_z[tz] + cpml->b_z[tz] * dvz_dz;

			fields->p[tid_global] = fields->p[tid_global] -
				config->delta_t * r_velocity * r_velocity * params->d_density[tid_global] * (
					dvx_dx + dvy_dy + dvz_dz + r_psi_vel_x + r_psi_vel_y + r_psi_vel_z
			);

			cpml->psi_vel_x[tid_global] = r_psi_vel_x;
			cpml->psi_vel_y[tid_global] = r_psi_vel_y;
			cpml->psi_vel_z[tid_global] = r_psi_vel_z;

			behind[3] = behind[2];
			behind[2] = behind[1];
			behind[1] = behind[0];
			behind[0] = front[0];
			front[0] = front[1];
			front[1] = front[2];
			front[2] = front[3];
			front[3] = fields->vel_z[tid_global_front];
		}
	}
}

__global__ void kernel_dvdxyz(PropagatorSetup *setup){
	
	PropagatorConfig *__restrict__ config = setup->propagator_config;
	PropagatorFields *__restrict__ fields = setup->propagator_fields;
	InversionParams *__restrict__ params = setup->inversion_params;
	CPML *__restrict__ cpml = setup->cpml;

	integer_t s_tx = threadIdx.x;
	integer_t s_ty = threadIdx.y;
	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;
	integer_t ty = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ real_t s_p[(2*STENCIL+TPBX)*(2*STENCIL+TPBY)];
	
	real_t front[4];
	real_t behind[4];
	real_t dp_dx, dp_dy, dp_dz;

	//load behind and front once
	for(integer_t l=0;l<STENCIL;l++){
		integer_t tid_global_behind = tx + (ty + l * config->ny) * config->nx;
		integer_t tid_global_front = tx + (ty + (l+STENCIL) * config->ny) * config->nx;
		front[l] = fields->p[tid_global_front];
		behind[STENCIL-1-l] = fields->p[tid_global_behind];
	}

	for(integer_t tz=STENCIL-1;tz<config->nz;tz++){

		integer_t tid_global = tx + (ty + tz * config->ny) * config->nx;
		integer_t tid_global_front = tx + (ty + (tz+STENCIL) * config->ny) * config->nx;

		real_t r_psi_x = setup->cpml->psi_x[tid_global];
		real_t r_psi_y = setup->cpml->psi_y[tid_global];
		real_t r_psi_z = setup->cpml->psi_z[tid_global];

		//copy center block inside shared memory
		if(tx < config->nx && ty < config->ny && tz < config->nz){
			s_p[s_tx+STENCIL + (s_ty+STENCIL)*(TPBX+2*STENCIL)] = fields->p[tid_global];
		}
		//copy left side halo inside shared memory. duplicating first STENCIL elements per x slice
		if(s_tx < STENCIL && tx >= STENCIL && ty < config->ny && tz < config->nz){
			s_p[s_tx + (s_ty+STENCIL)*(TPBX+2*STENCIL)] = fields->p[tid_global - STENCIL];
		}
		//copy right side halo inside shared memory. duplicating last STENCIL elements per x slice(+TPBX offset)
		if(s_tx < STENCIL && tx < (config->nx-TPBX) && ty < config->ny && tz < config->nz){
			s_p[s_tx+TPBX+STENCIL + (s_ty+STENCIL)*(TPBX+2*STENCIL)] = fields->p[tid_global + TPBX];
		}
		//back side halo copy inside shared memory
		if(tx < config->nx && s_ty < STENCIL && ty >= STENCIL && tz < config->nz){
			s_p[s_tx+STENCIL + s_ty*(TPBX+2*STENCIL)] = fields->p[tid_global - STENCIL*config->nx];
		}
		//front side halo copy inside shared memory
		if(tx < config->nx && s_ty < STENCIL && ty < (config->ny-TPBY) && tz < config->nz){
			s_p[s_tx+STENCIL + (s_ty+STENCIL+TPBY)*(TPBX+2*STENCIL)] = fields->p[tid_global + TPBY*config->nx];
		}
		__syncthreads();
	
		if(tx >= STENCIL && tx < (config->nx-STENCIL) && ty >= STENCIL && ty < (config->ny-STENCIL) && tz >= STENCIL && tz < (config->nz-STENCIL)){

			integer_t tid_shared = s_tx+STENCIL + (s_ty+STENCIL)*(2*STENCIL+TPBX);

			dp_dx = s_compute_stencil_backward(s_p, tid_shared, 1, config->delta_x);
			dp_dy = s_compute_stencil_backward(s_p, tid_shared, TPBX+2*STENCIL, config->delta_y);

			//update psi fields rather than in separate kernel call
			r_psi_x = r_psi_x * cpml->a_x[tx] + cpml->b_x[tx] * dp_dx;
			r_psi_y = r_psi_y * cpml->a_y[ty] + cpml->b_y[ty] * dp_dy;
		}

		if(tx > 2 && tx < (config->nx-4) && ty > 2 && ty < (config->ny-4) && tz > 2 && tz < (config->nz-4)){


			dp_dz = compute_stencil_backward_z(front, behind, config->delta_z);
			r_psi_z = r_psi_z * cpml->a_z[tz] + cpml->b_z[tz] * dp_dz;

			//update psi fields rather than in separate kernel call
			fields->vel_x[tid_global] = fields->vel_x[tid_global] -
			config->delta_t * (2.0 / params->d_density[tid_global]) * (dp_dx  + r_psi_x);

			fields->vel_y[tid_global] = fields->vel_y[tid_global] -
			config->delta_t * (2.0 / params->d_density[tid_global]) * (dp_dy  + r_psi_y);

			fields->vel_z[tid_global] = fields->vel_z[tid_global] -
			config->delta_t * (2.0 / params->d_density[tid_global]) * (dp_dz  + r_psi_z);

			cpml->psi_x[tid_global] = r_psi_x;
			cpml->psi_y[tid_global] = r_psi_y;
			cpml->psi_z[tid_global] = r_psi_z;

			behind[3] = behind[2];
			behind[2] = behind[1];
			behind[1] = behind[0];
			behind[0] = front[0];
			front[0] = front[1];
			front[1] = front[2];
			front[2] = front[3];
			front[3] = fields->p[tid_global_front];
			
		}
	}
}

