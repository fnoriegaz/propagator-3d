#ifndef PROPAGATOR_CONFIG_H
#define PROPAGATOR_CONFIG_H

#include "propagator_constants.h"

typedef struct PropagatorConfig{
	integer_t nx;
	integer_t ny;
	integer_t nz;

	integer_t timesamples;

	integer_t n_sources;
	integer_t n_receivers_x;
	integer_t n_receivers_y;
	integer_t receivers_depth;

	integer_t cpml_width;

	real_t delta_t;
	real_t delta_x;
	real_t delta_y;
	real_t delta_z;
	real_t r;
	real_t freq;
	real_t propagation_time;

}PropagatorConfig;

#endif
