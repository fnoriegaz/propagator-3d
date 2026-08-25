#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "propagator_config.h"

typedef struct PropagatorSetup PropagatorSetup;

PropagatorSetup *propagator_init(PropagatorConfig *propagator_config);

void propagate(PropagatorSetup *propagator_setup);

void propagator_finalize(PropagatorSetup *propagator_setup);

#endif
