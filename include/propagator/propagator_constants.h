#ifndef PROPAGATOR_CONSTANTS_H
#define PROPAGATOR_CONSTANTS_H

#ifdef USE_DOUBLE
	typedef double real_t;
#else
	typedef float real_t;
#endif

#ifdef USE_LONG_INT
	typedef long int integer_t;
#else
	typedef int integer_t;
#endif


#define PI 3.141592653589793
const real_t coeff[4]={1225.f/1024, 245.f/3072, 49.f/5120, 5.f/7168};


#endif

