#ifndef FDTD_H
#define FDTD_H

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif


__constant__ float coeff[4]={1225.f/1024, 245.f/3072, 49.f/5120, 5.f/7168};

// compute stencil "forward" direction. stride would define which dimenion is derivative being computed for
inline HOSTDEVICE compute_stencil_forward(float *field, int idx, int stride, float delta){
	float output = (coeff[0] * (field[idx] - field[idx-1*stride]) -
			coeff[1] * (field[idx+1*stride] - field[idx-2*stride]) +
			coeff[2] * (field[idx+2*stride] - field[idx-3*stride]) -
			coeff[3] * (field[idx+3*stride] - field[idx-4*stride])) / delta;
}

// compute stencil "backward" direction. 
inline HOSTDEVICE compute_stencil_backward(float *field, int idx, int stride, float delta){
	float output = (coeff[0] * (field[idx+1*stride] - field[idx]) -
			coeff[1] * (field[idx+2*stride] - field[idx-1*stride]) +
			coeff[2] * (field[idx+3*stride] - field[idx-2*stride]) -
			coeff[3] * (field[idx+4*stride] - field[idx-3*stride])) / delta;
}

#endif
