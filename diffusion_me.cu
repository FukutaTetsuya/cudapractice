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

//GPU functions-----------------------------------------------------------------

//Host functions----------------------------------------------------------------
void init_field(double *field_host, int n_host, int l_host) {
	int i;
	int j;
	double x;
	double y;
	double dx = l_host / (double)n_host;
	double dy = l_host / (double)n_host;
	double midst = l_host * 0.5;
	for(i = 0; i < n_host; i += 1) {
		x = (double)i * dx;
		for(j = 0; j < n_host; j += 1) {
			y = (double)j * dy;
			if((x > midst && y > midst) || (x < midst && y < midst)) {
				field_host[n_host * j + i] = 1.0;
			} else {
				field_host[n_host * j + i] = 0.0;
			}
		}
	}
}

int main(void) {
//variants----------------------------------------------------------------------
	int n_host;
	int n_square;
	int iteration;
	int n_blocks;
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
	n_square = N * N;
	l_host = L;
	dim_threads.x = NT;
	dim_threads.y = NT;
	dim_threads.z = 1;
	n_blocks = (int)(ceil((double)n_host / NT));
	iteration = M;

	cudaMemcpyToSymbol(n, &n_host, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(theta, &theta_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaHostAlloc((void **)&field_host[0], n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&field_host[1], n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&result_global_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&result_shared_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaMalloc((void **)&field_device[0], n_square * sizeof(double));
	cudaMalloc((void **)&field_device[1], n_square * sizeof(double));

//finalize----------------------------------------------------------------------
	cudaFreeHost(field_host[0]);
	cudaFreeHost(field_host[1]);
	cudaFreeHost(result_global_host);
	cudaFreeHost(result_shared_host);
	cudaFree(field_device[0]);
	cudaFree(field_device[1]);

	return 0;
}
