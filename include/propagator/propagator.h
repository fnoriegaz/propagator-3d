#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "propagator_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PropagatorSetup PropagatorSetup;

PropagatorSetup *propagator_init(PropagatorConfig *propagator_config, const char *config_file);

void propagate(PropagatorSetup *propagator_setup);

void propagator_finalize(PropagatorSetup *propagator_setup);

void custom_init(PropagatorSetup *setup);


#ifdef __cplusplus
}
#endif

#endif
