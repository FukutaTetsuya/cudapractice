#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NT 32//NT*NT threads per block
#define NB 8//blocks to use
#define N 600//N*N square matrix
__device__ const int n;

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
//initialize--------------------------------------------------------------------
	n_host = N;
	A_host = (double *)calloc(n_host * n_host, sizeof(double));
	B_host = (double *)calloc(n_host * n_host, sizeof(double));
	C_host = (double *)calloc(n_host * n_host, sizeof(double));
	//set variables---------------------------------------------------------
	init_matrix(A_host, n_host);
	init_matrix(B_host, n_host);

//finalize----------------------------------------------------------------------
	free(A_host);
	free(B_host);
	free(C_host);
	return 0;
}	

