#ifndef PROPAGATOR_STRUCTS_H
#define PROPAGATOR_STRUCTS_H

#include "../include/propagator/propagator_config.h"

typedef struct PropagatorFields{
	real_t *p;
	real_t *vel_x;
	real_t *vel_y;
	real_t *vel_z;
	real_t *lambda;
	real_t *lambda_x;
	real_t *lambda_y;
	real_t *lambda_z;
}PropagatorFields;

typedef struct InversionParams{
	real_t *h_velocity;
	real_t *h_density;
	real_t *d_velocity;
	real_t *d_density;
}InversionParams;

typedef struct SrcRecGeometry{
	real_t *h_source;
	real_t *d_gather;

	integer_t *src_positions;
	integer_t *rec_positions;
}SrcRecGeometry;

typedef struct CPML{
	real_t *psi_x;
	real_t *psi_y;
	real_t *psi_z;
	real_t *psi_vel_x;
	real_t *psi_vel_y;
	real_t *psi_vel_z;

	real_t *a_x;
	real_t *b_x;
	real_t *a_y;
	real_t *b_y;
	real_t *a_z;
	real_t *b_z;
}CPML;

typedef struct PropagatorSetup{
	PropagatorConfig *propagator_config;
	PropagatorFields *propagator_fields;
	InversionParams *inversion_params;
	SrcRecGeometry *sr_geometry;
	CPML *cpml;
}PropagatorSetup;

#endif
