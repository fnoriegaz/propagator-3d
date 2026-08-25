#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "../include/propagator/propagator.h"


int main(int argc, char *argv[]){

	const char * config_file = "/home/fabian/Documents/git/propagator-3d/src/init_config.par";

	PropagatorConfig propagator_config;
	PropagatorSetup *propagator_setup;

	propagator_setup = propagator_init(&propagator_config, config_file);

	custom_init(propagator_setup);

	propagate(propagator_setup);

	propagator_finalize(propagator_setup);

	printf("This is a check...\n");

	return 0;
}
