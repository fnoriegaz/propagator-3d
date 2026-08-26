#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "propagator_structs.h"
#include "../include/propagator/propagator.h"
#include "kernels.cuh"


void print_cuda_error(cudaError_t error, const char *filename, const char *function_name, int line){
	if(error != cudaSuccess){
		printf("in function %s, file %s, line %d, cuda error: %d...\n", function_name, filename, line, error);
		exit(-1);
	}
}

void custom_init(PropagatorSetup *setup){

	integer_t nx = setup->propagator_config->nx;
	integer_t ny = setup->propagator_config->ny;
	integer_t nz = setup->propagator_config->nz;
	integer_t timesamples = setup->propagator_config->timesamples;

	cudaError_t error;

	setup->sr_geometry->h_src_positions[0] = nx / 2;
	setup->sr_geometry->h_src_positions[1] = ny / 2;
	setup->sr_geometry->h_src_positions[2] = 48;

	for(integer_t t=0;t<timesamples;t++){
		real_t tt = (t * setup->propagator_config->delta_t) * (t * setup->propagator_config->delta_t);
		setup->sr_geometry->h_source[t] = (2.0 / (sqrt(3) * pow(PI,0.25))) * (1 - tt) * exp(-1 * tt / 2);
	}

	for(integer_t z=0;z<nz;z++){
		for(integer_t y=0;y<ny;y++){
			for(integer_t x=0;x<nx;x++){
				setup->inversion_params->h_velocity[x+nx*(y+ny*z)] = 1500.0;
				setup->inversion_params->h_density[x+nx*(y+ny*z)] = 2600.0;
			}
		}
	}

	cudaMemcpy(setup->sr_geometry->d_src_positions, setup->sr_geometry->h_src_positions, 3*sizeof(integer_t), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);
	cudaMemcpy(setup->sr_geometry->d_source, setup->sr_geometry->h_source, timesamples*sizeof(real_t), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);
	cudaMemcpy(setup->inversion_params->d_velocity, setup->inversion_params->h_velocity, nx*ny*nz*sizeof(real_t), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);
	cudaMemcpy(setup->inversion_params->d_density, setup->inversion_params->h_density, nx*ny*nz*sizeof(real_t), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);

}


void process_config_file(const char* config_file, PropagatorConfig *config){

	FILE *fid;
	fid = fopen(config_file, "r");
	if(fid == NULL){
		printf("file %s does not exist...\n",config_file);
		exit(-1);
	}

	char *line = NULL;
	ssize_t line_size;
	size_t n_bytes = 0;
	integer_t n_keys = 16;

	InitParamEntry param_entry_mapping[16] = {
		{"nx", &config->nx, TYPE_INTEGER},
		{"ny", &config->ny, TYPE_INTEGER},
		{"nz", &config->nz, TYPE_INTEGER},
		{"timesamples", &config->timesamples, TYPE_INTEGER},
		{"cpml_width", &config->cpml_width, TYPE_INTEGER},
		{"n_receivers_x", &config->n_receivers_x, TYPE_INTEGER},
		{"n_receivers_y", &config->n_receivers_y, TYPE_INTEGER},
		{"n_sources", &config->n_sources, TYPE_INTEGER},
		{"receivers_depth", &config->receivers_depth, TYPE_INTEGER},
		{"delta_x", &config->delta_x, TYPE_FLOAT},
		{"delta_y", &config->delta_y, TYPE_FLOAT},
		{"delta_z", &config->delta_z, TYPE_FLOAT},
		{"delta_t", &config->delta_t, TYPE_FLOAT},
		{"max_vel", &config->max_vel, TYPE_FLOAT},
		{"r", &config->r, TYPE_FLOAT},
		{"freq", &config->freq, TYPE_FLOAT}
	};

	//read the init par file until EOF. aka getline returns -1
	while((line_size = getline(&line, &n_bytes, fid)) != -1){

		char *data_string = (char *)calloc(16,sizeof(char));
		memset(data_string,0x00, 16);
		int data_str_count = 0;

		for(int k=0;k<n_keys;k++){
			int key_len = strlen((const char*)param_entry_mapping[k].param_name);
			int compared = strncmp((const char*)line, param_entry_mapping[k].param_name,key_len);
			if(compared == 0){
				int count = key_len;
				while(count < line_size){
					if(line[count] != ' '){
						data_string[data_str_count] = line[count];
						data_str_count++;

						if(data_str_count > 16) break; //break the while and then what??
					}
					count++;
				}
				//i believe in the atoi and atof implementations
				if(param_entry_mapping[k].param_type == TYPE_INTEGER){
					*(integer_t *)param_entry_mapping[k].param = atoi(data_string);
				}
				else{
					*(real_t *)param_entry_mapping[k].param = atof(data_string);
				}
			}
		}

		free(data_string);
	}
	free(line);
	config->propagation_time = config->timesamples * config->delta_t;
	fclose(fid);
}

PropagatorSetup *propagator_init(PropagatorConfig *config, const char *config_file){
	
	process_config_file(config_file,config);

	PropagatorSetup *propagator_setup = (PropagatorSetup *)calloc(1, sizeof(PropagatorSetup));
	PropagatorFields *propagator_fields = (PropagatorFields *)calloc(1, sizeof(PropagatorFields));
	InversionParams *inversion_params = (InversionParams *)calloc(1, sizeof(InversionParams));
	SrcRecGeometry *sr_geometry = (SrcRecGeometry *)calloc(1, sizeof(SrcRecGeometry));
	CPML *cpml = (CPML *)calloc(1, sizeof(CPML));
	
	cudaMalloc((void **)&propagator_fields->p, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->p, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->vel_x, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->vel_y, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->vel_z, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->vel_z, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->lambda, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->lambda_x, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->lambda_y, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&propagator_fields->lambda_z, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(propagator_fields->lambda_z, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	
	cudaMalloc((void **)&inversion_params->d_velocity, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(inversion_params->d_velocity, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&inversion_params->d_density, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(inversion_params->d_density, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	inversion_params->h_velocity = (real_t *)calloc(config->nx*config->ny*config->nz, sizeof(real_t));
	inversion_params->h_density = (real_t *)calloc(config->nx*config->ny*config->nz, sizeof(real_t));

	sr_geometry->h_source = (real_t *)calloc(config->timesamples, sizeof(real_t));
	sr_geometry->h_gather = (real_t *)calloc(config->n_receivers_x*config->n_receivers_y*config->timesamples, sizeof(real_t));
	sr_geometry->h_src_positions = (integer_t *)calloc(3*config->n_sources, sizeof(integer_t));
	sr_geometry->h_rec_positions = (integer_t *)calloc(3*config->n_receivers_x*config->n_receivers_y, sizeof(integer_t));
	cudaMalloc((void **)&sr_geometry->d_source, config->timesamples*sizeof(real_t));
	cudaMemset(sr_geometry->d_source, 0.0, config->timesamples*sizeof(real_t));
	cudaMalloc((void **)&sr_geometry->d_gather, config->n_receivers_x*config->n_receivers_y*config->timesamples*sizeof(real_t));
	cudaMemset(sr_geometry->d_gather, 0.0, config->n_receivers_x*config->n_receivers_y*config->timesamples*sizeof(real_t));
	cudaMalloc((void **)&sr_geometry->d_src_positions, 3*config->n_sources*sizeof(integer_t));
	cudaMemset(sr_geometry->d_src_positions, 0, 3*config->n_sources*sizeof(integer_t));
	cudaMalloc((void **)&sr_geometry->d_rec_positions, 3*config->n_receivers_x*config->n_receivers_y*sizeof(integer_t));
	cudaMemset(sr_geometry->d_rec_positions, 0, 3*config->n_receivers_x*config->n_receivers_y*sizeof(integer_t));

	cudaMalloc((void **)&cpml->a_x, config->nx*sizeof(real_t));
	cudaMemset(cpml->a_x, 0.0, config->nx*sizeof(real_t));
	cudaMalloc((void **)&cpml->b_x, config->nx*sizeof(real_t));
	cudaMemset(cpml->b_x, 0.0, config->nx*sizeof(real_t));
	cudaMalloc((void **)&cpml->a_y, config->ny*sizeof(real_t));
	cudaMemset(cpml->a_y, 0.0, config->ny*sizeof(real_t));
	cudaMalloc((void **)&cpml->b_y, config->ny*sizeof(real_t));
	cudaMemset(cpml->b_y, 0.0, config->ny*sizeof(real_t));
	cudaMalloc((void **)&cpml->a_z, config->nz*sizeof(real_t));
	cudaMemset(cpml->a_z, 0.0, config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->b_z, config->nz*sizeof(real_t));
	cudaMemset(cpml->b_z, 0.0, config->nz*sizeof(real_t));

	cudaMalloc((void **)&cpml->psi_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_x, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->psi_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_y, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->psi_z, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_z, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->psi_vel_x, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_vel_x, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->psi_vel_y, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_vel_y, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMalloc((void **)&cpml->psi_vel_z, config->nx*config->ny*config->nz*sizeof(real_t));
	cudaMemset(cpml->psi_vel_z, 0.0, config->nx*config->ny*config->nz*sizeof(real_t));

	propagator_setup->propagator_config = config;
	propagator_setup->propagator_fields = propagator_fields;
	propagator_setup->inversion_params = inversion_params;
	propagator_setup->sr_geometry = sr_geometry;
	propagator_setup->cpml = cpml;

	return propagator_setup;
}


void propagator_finalize(PropagatorSetup *setup){
	cudaFree(setup->propagator_fields->p);
	cudaFree(setup->propagator_fields->vel_x);
	cudaFree(setup->propagator_fields->vel_y);
	cudaFree(setup->propagator_fields->vel_z);
	cudaFree(setup->propagator_fields->lambda);
	cudaFree(setup->propagator_fields->lambda_x);
	cudaFree(setup->propagator_fields->lambda_y);
	cudaFree(setup->propagator_fields->lambda_z);

	cudaFree(setup->sr_geometry->d_source);
	cudaFree(setup->sr_geometry->d_gather);
	cudaFree(setup->sr_geometry->d_rec_positions);
	cudaFree(setup->sr_geometry->d_src_positions);

	cudaFree(setup->inversion_params->d_velocity);
	cudaFree(setup->inversion_params->d_density);

	cudaFree(setup->cpml->a_x);
	cudaFree(setup->cpml->a_y);
	cudaFree(setup->cpml->a_z);
	cudaFree(setup->cpml->b_x);
	cudaFree(setup->cpml->b_y);
	cudaFree(setup->cpml->b_z);
	cudaFree(setup->cpml->psi_x);
	cudaFree(setup->cpml->psi_y);
	cudaFree(setup->cpml->psi_z);
	cudaFree(setup->cpml->psi_vel_x);
	cudaFree(setup->cpml->psi_vel_y);
	cudaFree(setup->cpml->psi_vel_z);
}


void propagate(PropagatorSetup *setup){

	cudaError_t error;
	integer_t nx = setup->propagator_config->nx;
	integer_t ny = setup->propagator_config->ny;
	integer_t nz = setup->propagator_config->nz;
	integer_t timesamples = setup->propagator_config->timesamples;

	//variables for manual testing
	real_t test_p;
	integer_t src_pos_offset = nx / 2 + nx * (ny/2 + ny*32);
	integer_t src_idx = 0;
	real_t *shotgather = (real_t *)calloc(nx*ny*timesamples,sizeof(real_t));

	dim3 tpb(TPBX,TPBY,TPBZ);
	dim3 bpg((nx+TPBX-1)/TPBX,(ny+TPBY-1)/TPBY,(nz+TPBZ-1)/TPBZ);

	kernel_cpml<<<(nx+TPBX-1)/TPBX,TPBX>>>(setup->cpml->a_x, setup->cpml->b_x, nx, setup->propagator_config->freq,
				 setup->propagator_config->r, setup->propagator_config->delta_x, setup->propagator_config->delta_t,
				 setup->propagator_config->max_vel, setup->propagator_config->cpml_width);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);

	kernel_cpml<<<(ny+TPBY-1)/TPBY,TPBY>>>(setup->cpml->a_y, setup->cpml->b_y, ny, setup->propagator_config->freq,
				 setup->propagator_config->r, setup->propagator_config->delta_y, setup->propagator_config->delta_t,
				 setup->propagator_config->max_vel, setup->propagator_config->cpml_width);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);

	kernel_cpml<<<(nz+TPBZ-1)/TPBZ,TPBZ>>>(setup->cpml->a_z, setup->cpml->b_z, nz, setup->propagator_config->freq,
				 setup->propagator_config->r, setup->propagator_config->delta_z, setup->propagator_config->delta_t,
				 setup->propagator_config->max_vel, setup->propagator_config->cpml_width);
	error = cudaGetLastError();
	print_cuda_error(error,__FILE__,__func__,__LINE__);

	for(int t=0;t<timesamples;t++){

		kernel_add_source<<<1,1>>>(setup, src_idx, t);
		error = cudaGetLastError();
		print_cuda_error(error,__FILE__,__func__,__LINE__);

		kernel_dpdt<<<bpg,tpb>>>(setup);
		error = cudaGetLastError();
		print_cuda_error(error,__FILE__,__func__,__LINE__);

		kernel_dvdxyz<<<bpg,tpb>>>(setup);
		error = cudaGetLastError();
		print_cuda_error(error,__FILE__,__func__,__LINE__);

		cudaMemcpy(&test_p, setup->propagator_fields->p+src_pos_offset, 1*sizeof(real_t), cudaMemcpyDeviceToHost);
		error = cudaGetLastError();
		print_cuda_error(error,__FILE__,__func__,__LINE__);

		//manually copy slices to simulate the shotgather information
		cudaMemcpy(shotgather+t*nx*ny,setup->propagator_fields->p+32*nx*ny, nx*ny*sizeof(real_t),cudaMemcpyDeviceToHost);

	}

	FILE *fid;
	fid = fopen("shotgather.bin","wb+");
	fwrite(shotgather,nx*ny*timesamples,sizeof(real_t),fid);
	fclose(fid);

	free(shotgather);

}



