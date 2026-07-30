#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "propagator_structs.h"
#include "../include/propagator/propagator.h"


PropagatorSetup *propagator_init(PropagatorConfig *config){
	
	cudaError_t error;

	PropagatorSetup *propagator_setup = (PropagatorSetup *)calloc(1, sizeof(PropagatorSetup));
	PropagatorFields *propagator_fields = (PropagatorFields *)calloc(1, sizeof(PropagatorFields));
	InversionParams *inversion_params = (InversionParams *)calloc(1, sizeof(InversionParams));
	SrcRecGeometry *sr_geometry = (SrcRecGeometry *)calloc(1, sizeof(sr_geometry));
	CPML *cpml = (CPML *)calloc(1, sizeof(cpml));

	config->nx = 256;
	config->ny = 256;
	config->nz = 256;

	config->timesamples = 3000;

	config->n_sources = 1;
	config->n_receivers_x = 256;
	config->n_receivers_y = 1;
	config->receivers_depth = 50;

	config->cpml_width = 32;

	config->delta_t = 1e-3;
	config->delta_x = 25.0;
	config->delta_y = 25.0;
	config->delta_z = 25.0;
	config->r = 1e-4;
	config->freq = 3.0;
	config->propagation_time = config->timesamples * config->delta_t;
	
	cudaMalloc((void **)&propagator_fields->p, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_z, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_z, config->nx*config->ny*config->nz*sizeof(real_t));
	
	cudaMalloc((void **)&inversion_params->d_velocity, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&inversion_params->d_density, config->nx*config->ny*config->nz*sizeof(real_t));
	inversion_params->h_velocity = (real_t *)calloc(config->nx*config->ny*config->nz, sizeof(real_t));
	inversion_params->h_density = (real_t *)calloc(config->nx*config->ny*config->nz, sizeof(real_t));

	sr_geometry->h_source = (real_t *)calloc(config->timesamples, sizeof(real_t));
	cudaMalloc((void **)&sr_geometry->d_gather, config->n_receivers_x*config->n_receivers_y*config->timesamples*sizeof(real_t));

	//cudaMalloc((void **)&cpml->psi_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->a_x, config->nx*sizeof(real_t));
	cudaMalloc((void **)&cpml->b_x, config->nx*sizeof(real_t));
	cudaMalloc((void **)&cpml->a_y, config->ny*sizeof(real_t));
	//cudaMalloc((void **)&cpml->b_y, config->ny*sizeof(real_t));
	//cudaMalloc((void **)&cpml->a_z, config->nz*sizeof(real_t));
	//cudaMalloc((void **)&cpml->b_z, config->nz*sizeof(real_t));

	//cudaMalloc((void **)&cpml->psi_z, sizeof(real_t));
	//cudaMalloc((void **)&cpml->psi_x, config->nx*config->ny*config->nz*sizeof(real_t));
	//cudaMalloc((void **)&cpml->psi_y, config->nx*config->ny*config->nz*sizeof(real_t));
	//cudaMalloc((void **)&cpml->psi_vel_x, config->nx*config->ny*config->nz*sizeof(real_t));
	//cudaMalloc((void **)&cpml->psi_vel_y, config->nx*config->ny*config->nz*sizeof(real_t));
	//cudaMalloc((void **)&cpml->psi_vel_z, config->nx*config->ny*config->nz*sizeof(real_t));
	
	propagator_setup->propagator_config = config;
	propagator_setup->propagator_fields = propagator_fields;
	propagator_setup->inversion_params = inversion_params;
	propagator_setup->sr_geometry = sr_geometry;
	propagator_setup->cpml = cpml;

	return propagator_setup;
}


int main(int argc, char *argv[]){

	PropagatorConfig propagator_config;
	PropagatorSetup *propagator_setup = (PropagatorSetup *)calloc(1, sizeof(PropagatorSetup));
	propagator_setup = propagator_init(&propagator_config);


	free(propagator_setup);
	return 0;
}
