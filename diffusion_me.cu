#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*  square root of number of threads in a block (the number of threads in a block is NT^2)  */
#define NT 32
/*  length of the target domain  */
#define L 10.0
/*  number of division for the discretization of the target domain  */
#define N 256
/*  dimensionless time step size (theta = D * dt / dx^2)  */
#define THETA 0.1
/*  number of iterations  */
#define M 2000


/*  constants on a GPU  */
__device__ __constant__ int n;
__device__ __constant__ float theta;

//GPU functions----------------------------------------------------------------


int main(void) {
	int n_host;
	int n_square;
	int m;
	double l_host;
	double theta_host;
	dim3 dim_threads;
	double *field_host[2];
	double *field_device[2];
	double *result_global_host;
	double *result_shared_host;
	FILE *file_write;

//initialize--------------------------------------------------------------------
	n_host = N;
	l_host = L;
	dim_threads.x = NT;
	dim_threads.y = NT;
	dim_threads.z = 1;
	n_blocks = (int)(ceil((double)n_host / NT));

	return 0;
}
