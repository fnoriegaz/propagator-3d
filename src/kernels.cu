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

	integer_t tx = threadIdx.x + blockIdx.x * blockDim.x;
	integer_t ty = threadIdx.y + blockIdx.y * blockDim.y;
	integer_t tz = threadIdx.z + blockIdx.z * blockDim.z;
	integer_t tid_global = tx + (ty + tz * setup->propagator_config->ny) * setup->propagator_config->nx;

	real_t r_velocity = setup->inversion_params->d_velocity[tid_global];
	real_t r_psi_vel_x = setup->cpml->psi_vel_x[tid_global];
	real_t r_psi_vel_y = setup->cpml->psi_vel_y[tid_global];
	real_t r_psi_vel_z = setup->cpml->psi_vel_z[tid_global];

	integer_t s_tx = threadIdx.x;
	integer_t s_ty = threadIdx.y;
	integer_t s_tz = threadIdx.z;
	
	real_t dvx_dx = 0.;
	real_t dvy_dy = 0.;
	real_t dvz_dz = 0.;
	__shared__ real_t s_vel_x[(TPBX+8)*TPBY*TPBZ];
	__shared__ real_t s_vel_y[TPBX*(TPBY+8)*TPBZ];
	__shared__ real_t s_vel_z[TPBX*TPBY*(TPBZ+8)];

	//copy center block inside shared memory
	if(tx < setup->propagator_config->nx && ty < setup->propagator_config->ny && tz < setup->propagator_config->nz){
		s_vel_x[s_tx+STENCIL + s_ty*(TPBX+2*STENCIL) + s_tz*(TPBX+2*STENCIL)*TPBY] = setup->propagator_fields->vel_x[tid_global];
		s_vel_y[s_tx + (s_ty+STENCIL)*TPBX + s_tz*TPBX*(TPBY+2*STENCIL)] = setup->propagator_fields->vel_y[tid_global];
		s_vel_z[s_tx + s_ty*TPBX + (s_tz+STENCIL)*TPBX*TPBY] = setup->propagator_fields->vel_z[tid_global];
	}
	//copy left side halo inside shared memory. duplicating first STENCIL elements per x slice
	if(s_tx < STENCIL && tx >= STENCIL && ty < setup->propagator_config->ny && tz < setup->propagator_config->nz){
		s_vel_x[s_tx + s_ty*(TPBX+2*STENCIL) + s_tz*(TPBX+2*STENCIL)*TPBY] = setup->propagator_fields->vel_x[tid_global - STENCIL];
	}
	//copy right side halo inside shared memory. duplicating last STENCIL elements per x slice(+TPBX offset)
	if(s_tx < STENCIL && tx < (setup->propagator_config->nx-TPBX) && ty < setup->propagator_config->ny && tz < setup->propagator_config->nz){
		s_vel_x[s_tx + s_ty*(TPBX+2*STENCIL) + s_tz*(TPBX+2*STENCIL)*TPBY] = setup->propagator_fields->vel_x[tid_global + TPBX];
	}
	//back side halo copy inside shared memory
	if(tx < setup->propagator_config->nx && s_ty < STENCIL && ty >= STENCIL && tz < setup->propagator_config->nz){
		s_vel_y[s_tx + s_ty*TPBX + s_tz*TPBX*(TPBY+2*STENCIL)] = setup->propagator_fields->vel_y[tid_global - STENCIL*setup->propagator_config->nx];
	}
	//front side halo copy inside shared memory
	if(tx < setup->propagator_config->nx && s_ty < STENCIL && ty < (setup->propagator_config->ny-TPBY) && tz < setup->propagator_config->nz){
		s_vel_y[s_tx + s_ty*TPBX + s_tz*TPBX*(TPBY+2*STENCIL)] = setup->propagator_fields->vel_y[tid_global + TPBY*setup->propagator_config->nx];
	}
	//bottom side hallo copy inside shared memory
	if(tx < setup->propagator_config->nx && ty < setup->propagator_config->ny && s_tz < STENCIL && tz >= STENCIL){
		s_vel_z[s_tx + s_ty*TPBX + (s_tz+STENCIL)*TPBX*TPBY] = setup->propagator_fields->vel_z[tid_global - STENCIL*setup->propagator_config->nx*setup->propagator_config->ny];
	}
	//top side hallo copy inside shared memory
	if(tx < setup->propagator_config->nx && ty < setup->propagator_config->ny && s_tz < STENCIL && tz < (setup->propagator_config->nz-TPBZ)){
		s_vel_z[s_tx + s_ty*TPBX + (s_tz+STENCIL)*TPBX*TPBY] = setup->propagator_fields->vel_z[tid_global +TPBZ*setup->propagator_config->nx*setup->propagator_config->ny];
	}
	__syncthreads();

	if(tx >= STENCIL && tx < (setup->propagator_config->nx-STENCIL) && ty >= STENCIL && ty < (setup->propagator_config->ny-STENCIL) && tz >= STENCIL && tz < (setup->propagator_config->nz-STENCIL)){

		integer_t tid_shared_x = s_tx + STENCIL + s_ty*(2*STENCIL+TPBX) + s_tz*(2*STENCIL+TPBX)*TPBY;
		integer_t tid_shared_y = s_tx + (s_ty+STENCIL)*TPBX + s_tz*TPBX*(2*STENCIL+TPBY);
		integer_t tid_shared_z = s_tx + s_ty*TPBX + (s_tz+STENCIL)*TPBX*TPBY;

		dvx_dx = s_compute_stencil_forward(s_vel_x, tid_shared_x, 1, setup->propagator_config->delta_x);
		dvy_dy = s_compute_stencil_forward(s_vel_y, tid_shared_y, TPBX, setup->propagator_config->delta_y);
		dvz_dz = s_compute_stencil_forward(s_vel_z, tid_shared_z, TPBX*TPBY, setup->propagator_config->delta_z);

		//update psi fields rather than in separate kernel call
		r_psi_vel_x = r_psi_vel_x * setup->cpml->a_x[tx] + setup->cpml->b_x[tx] * dvx_dx;
		r_psi_vel_y = r_psi_vel_y * setup->cpml->a_y[ty] + setup->cpml->b_y[ty] * dvy_dy;
		r_psi_vel_z = r_psi_vel_z * setup->cpml->a_z[tz] + setup->cpml->b_z[tz] * dvz_dz;
	}

	if(tx > 3 && tx < (setup->propagator_config->nx-3) && ty > 3 && ty < (setup->propagator_config->ny-3) && tz > 3 && tz < (setup->propagator_config->nz-3)){
		setup->propagator_fields->p[tid_global] = setup->propagator_fields->p[tid_global] -
			setup->propagator_config->delta_t * r_velocity * r_velocity * setup->inversion_params->d_density[tid_global] * (
				dvx_dx + dvy_dy + dvz_dz + r_psi_vel_x + r_psi_vel_y + r_psi_vel_z
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
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * (dp_dx  + r_psi_x);

		setup->propagator_fields->vel_y[tid_global] = setup->propagator_fields->vel_y[tid_global] -
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * (dp_dy  + r_psi_y);

		setup->propagator_fields->vel_z[tid_global] = setup->propagator_fields->vel_z[tid_global] -
			setup->propagator_config->delta_t * (2.0 / setup->inversion_params->d_density[tid_global]) * (dp_dz  + r_psi_z);

		setup->cpml->psi_x[tid_global] = r_psi_x;
		setup->cpml->psi_y[tid_global] = r_psi_y;
		setup->cpml->psi_z[tid_global] = r_psi_z;

	}
}

