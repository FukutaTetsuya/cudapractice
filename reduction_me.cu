#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include<unistd.h>

#define DEVICE_ID 0

/*  constants for the number of threads and the integration domain  */
/*  number of threads in a block, 2^n  */
#define NT 1024
/*  number of blocks in a grid, 2^n  */
#define NB 16
/*  length of the target domain  */
#define L 10.0
/*  number of division for the discretization of the target domain, 2^n  */
#define N 1024

/*  constants on a GPU  */
__device__ __constant__ int n;
__device__ __constant__ double l;

//host functions-----------------------------------------------
double calculate_reference(int n, double l) {
	int i;
	int j;
	double d = l / (double)n;
	double x;
	double y;
	double sum = 0.0;
	for(i = 0; i < n; i += 1) {
		x = d * ((double)i - (double)n / 2.0);
		for(j = 0; j < n; j += 1) {
			y = d * ((double)j - (double)n / 2.0);
			sum += exp(- x * x - y * y);
		}
	}
	return sum * d * d;
}
//device functions---------------------------------------------
__global__ void calculate_each_point(double *device_double) {
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i;
	double x;
	double y;	
	for(i = global_id; i < n * n; i += NT * NB) {
		x = (l / (double)n) * ((double)(i % n) - (double)n / 2.0);
		y = (l / (double)n) * ((double)(i / n) - (double)n / 2.0);
		device_double[i] = exp(- x * x - y * y);
	}
}
//-------------------------------------------------------------
__global__ void reduce_array_global_memory(double *device_double, double *device_double_reduced, int dim_array) {
	int i;
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = global_id; i < dim_array; i += NT * NB) {
		if(i < dim_array / 2) {
			device_double_reduced[i] = device_double[i] + device_double[i + dim_array / 2];
		}	
	}
}
//-------------------------------------------------------------
__global__ void reduce_array_shared_memory(double *device_double, double *device_double_reduced, int dim_array) {
	__shared__ double device_shared_double[NT];
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int block_id = blockIdx.x;
	int local_id = threadIdx.x;
	int i;
	int j;
	for(i = global_id; i < dim_array; i += NB * NT) {
		device_shared_double[local_id] = device_double[i];
		__syncthreads();

		for(j = NT / 2; j > 0; j = j / 2) {
			if((local_id < j) && (local_id + j < dim_array)) {
				device_shared_double[local_id] += device_shared_double[local_id + j]; 
			}
		__syncthreads();
		}

		if(local_id == 0) {
			device_double_reduced[block_id] = device_shared_double[0];
		}
		__syncthreads();
		block_id += NB;
	}
}
//main---------------------------------------------------------
int main(void) {
	int host_n = N;
	double *device_double[2];
	double host_l = L;
	double host_sum;
	double *host_double;//DBG
	int i;
	int i_temp;
	int j;
	int k;

//initialize---------------------------------------------------
	cudaSetDevice(DEVICE_ID);
	cudaMemcpyToSymbol(n, &host_n, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(l, &host_l, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&device_double[0], N * N * sizeof(double));
	cudaMalloc((void **)&device_double[1], N * N * sizeof(double));
	host_double = (double *)calloc(N * N, sizeof(double));
//tasks--------------------------------------------------------
	//calculate on gpu w/o shared memory-------------------
	calculate_each_point<<<NB, NT>>>(device_double[0]);
	cudaDeviceSynchronize();
	i = 0;
	j = 1;
	for(k = N * N; k > 1; k = k / 2 + k % 2) { 	
		reduce_array_global_memory<<<NB, NT>>>(device_double[i], device_double[j], k);
		cudaDeviceSynchronize();
		i_temp = i;
		i = j;
		j = i_temp;
	}
	cudaMemcpy(&host_sum, device_double[i], sizeof(double), cudaMemcpyDeviceToHost);
	host_sum = host_sum * host_l * host_l / (double)host_n / (double)host_n;
	printf("calc on gpu w/o shared mem:%f\n", host_sum);
	//calculate on gpu with shared memory------------------
	calculate_each_point<<<NB, NT>>>(device_double[0]);
	cudaDeviceSynchronize();
	i = 0;
	j = 1;
	for(k = N * N; k > 1; k = k / NT) {
		reduce_array_shared_memory<<<NB, NT>>>(device_double[i], device_double[j], k);
		cudaDeviceSynchronize();
		i_temp = i;
		i = j;
		j = i_temp;
	}
	cudaMemcpy(&host_sum, device_double[i], sizeof(double), cudaMemcpyDeviceToHost);
	host_sum = host_sum * host_l * host_l / (double)host_n / (double)host_n;
	printf("calc on gpu with shared mem:%f\n", host_sum);

	//calculate on cpu-------------------------------------
	host_sum = calculate_reference(host_n, host_l);
	printf("all done on cpu:%f\n", host_sum);
//finalize-----------------------------------------------------
	cudaFree(device_double[0]);
	cudaFree(device_double[1]);
	cudaDeviceReset();
	free(host_double);

//work in host-------------------------------------------------

	return 0;
}


