#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NT 32//NT*NT threads per block
#define NB 8//blocks to use
#define N 600//N*N square matrix
__device__ __constant__ int n;//n = N

//GPU functions-----------------------------------------------------------------
__global__ void matrix_product_global(double *A_device, double *B_device, double *C_device) {
	//1 thread 1 C element
	int i_global, j_global;
	int i, j, k;
	double temp;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	j_global = threadIdx.y;

	for(i = i_global; i < n; i += NB * NT) {
		for(j = j_global; j < n; j += NT) {
			temp = 0.0;
			for(k = 0; k < n; k += 1) {
				temp += A_device[i * n + k] * B_device[k * n + j];
			}
			C_device[i * n + j] = temp;
		}
	}
}

//Host functions----------------------------------------------------------------
void init_matrix(double *matrix, int n_host) {
	int i;
	int j;
	for(i = 0; i < n_host; i += 1) {
		for(j = 0; j < n_host; j += 1) {
			matrix[i * n_host + j] = (double)rand() / (double)RAND_MAX;
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
				temporal += A[i * n_host + k] * B[k * n_host + j];
			}
			C[i * n_host + j] = temporal;
		}
	}
}

double calc_deviation(double *M1, double *M2, int n_host) {
	int i;
	double devi = 0.0;
	for(i = 0; i < n_host; i += 1) {
		devi += (M1[i] - M2[i]) * (M1[i] - M2[i]);
	}
	return devi;
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
	//constants-------------------------------------------------------------
	n_host = N;
	n_square = n_host * n_host;
	dim_threads.x = NT;
	dim_threads.y = NT;
	dim_threads.z = 1;
	cudaMemcpyToSymbol(n, &n_host, sizeof(int), 0, cudaMemcpyHostToDevice);

	//allocate--------------------------------------------------------------
	cudaHostAlloc((void **)&A_host, n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&B_host, n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&C_host, n_square * sizeof(double), cudaHostAllocMapped);
	result_host = (double *)calloc(n_square, sizeof(double));
	result_global = (double *)calloc(n_square, sizeof(double));

	cudaMalloc((void **)&A_device, n_square * sizeof(double));
	cudaMalloc((void **)&B_device, n_square * sizeof(double));
	cudaMalloc((void **)&C_device, n_square * sizeof(double));

	//init matrix------------------------------------------------------------
	init_matrix(A_host, n_host);
	init_matrix(B_host, n_host);
	cudaMemcpy(A_device, A_host, n_square * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, n_square * sizeof(double), cudaMemcpyHostToDevice);

//calculate---------------------------------------------------------------------
	//host------------------------------------------------------------------
	start = clock();
	matrix_product_host(A_host, B_host, C_host, n_host);
	end = clock();
	memcpy(result_host, C_host, n_square * sizeof(double));
	printf("%d [ms]\n", (int)(1000*(end - start)/CLOCKS_PER_SEC));

	//global----------------------------------------------------------------
	start = clock();
	matrix_product_global<<<NB, dim_threads>>>(A_device, B_device, C_device);
	cudaDeviceSynchronize();
	end = clock();
	cudaMemcpy(result_global, C_device, n_square * sizeof(double), cudaMemcpyDeviceToHost);
	printf("%d [ms]\n", (int)(1000*(end - start)/CLOCKS_PER_SEC));

//check the answers-------------------------------------------------------------
	printf("check\n");
	printf("global:%f\n", calc_deviation(result_host, result_global, n_host));

//finalize----------------------------------------------------------------------
	cudaFreeHost(A_host);
	cudaFreeHost(B_host);
	cudaFreeHost(C_host);
	free(result_host);

	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);
	return 0;
}	

