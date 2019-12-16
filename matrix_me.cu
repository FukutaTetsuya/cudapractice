#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NT 32//NT*NT threads per block
#define NB 8//blocks to use
#define N 600//N*N square matrix
__device__ __constant__ int n;

//GPU functions-----------------------------------------------------------------
//Host functions----------------------------------------------------------------
void init_matrix(double *matrix, int n_host) {
	int i;
	int j;
	for(i = 0; i < n_host; i += 1) {
		for(j = 0; j < n_host; j += 1) {
			matrix[i + n_host * j] = 1.0;
		}
	}
}
void matrix_product_host(double *A, double *B, double *C, int n_host) {
	int i;
	int j;
	int k;
	double temporal;
	for(i = 0; i < n_host; i += 1) {
		for(j = 0; j < n_host; j += 1) {
			//ij loop for each component of matrix C
			temporal = 0.0;
			for(k = 0; k < n_host; k += 1) {
				temporal += A[i + n_host * k] * B[k + n_host * j];
			}
			C[i + n_host * j] = temporal;
		}
	}
}

int main(void) {
	int n_host;
	int n_square;
	dim3 dim_threads;
	double *A_device;
	double *B_device;
	double *C_device;
	double *A_host;
	double *B_host;
	double *C_host;
	double *result_host;
	double *result_global;
	double *result_shared;
	clock_t start, end;
//initialize--------------------------------------------------------------------
	n_host = N;
	n_square = n_host * n_host;
	cudaHostAlloc((void **)&A_host, n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&B_host, n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&C_host, n_square * sizeof(double), cudaHostAllocMapped);
	//set variables---------------------------------------------------------
	cudaMemcpyToSymbol(n, &n_host, sizeof(int), 0, cudaMemcpyHostToDevice);
	init_matrix(A_host, n_host);
	init_matrix(B_host, n_host);

//calculate---------------------------------------------------------------------
	//host------------------------------------------------------------------
	start = clock();
	matrix_product_host(A_host, B_host, C_host, n_host);
	end = clock();
	printf("%d [ms]\n", (int)(1000*(end - start)/CLOCKS_PER_SEC));

//finalize----------------------------------------------------------------------
	cudaFreeHost(A_host);
	cudaFreeHost(B_host);
	cudaFreeHost(C_host);
	return 0;
}	

