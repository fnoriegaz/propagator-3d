#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "propagator_config.h"

typedef struct PropagatorSetup PropagatorSetup;

PropagatorSetup *propagator_init(PropagatorConfig *propagator_config);

void propagator_forward(PropagatorSetup *propagator_setup);

void propagator_end(PropagatorSetup *propagator_setup);

#endif
