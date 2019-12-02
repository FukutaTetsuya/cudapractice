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
__device__ __constant__ double theta;

//GPU functions-----------------------------------------------------------------
__global__ void diffusion_global(double *field_device, double *field_device_new) {
	int i_global;
	int j_global;
	int i_top, i_bottom;
	int j_right, j_left;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	if(i_global < n) {
		i_top = (i_global + 1) % n;
		i_bottom = (i_global - 1 + n) % n;
		for(j_global = threadIdx.y; j_global < n; j_global += NT) {
			j_right = (j_global + 1) % n;
			j_left = (j_global - 1 + n) % n;
			field_device_new[i_global * n + j_global] = (1.0 - 4.0 * theta) * field_device[i_global * n + j_global]
				+ theta * (field_device[i_top * n + j_global] + field_device[i_bottom * n + j_global]
					      + field_device[i_global * n + j_right] + field_device[i_global * n + j_left]);
		}
	}
}

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

void flip_ij(int *i, int *j) {
	int i_temp;
	i_temp = *i;
	*i = *j;
	*j = i_temp;
}

void print_field(FILE *file_write, double *field, int n, double l) {
	int i;
	int j;
	double x;
	double y;
	double d = l/(double)n;
	for(i = 0; i < N; i += 1) {
		y = (double)i * d;
		for(j = 0; j < N; j += 1) {
			x = (double)j * d;
			fprintf(file_write, "%f %f %f\n", x, y, field[i * n + j]);
		}
	}
}

void diffusion_host(double *field_host, double *field_host_new, int n_host, double theta_host) {
	int i;
	int j;
	int i_top, i_bottom;
	int j_right, j_left;

	for(i = 0; i < n_host; i += 1) {
		i_top = (i + 1) % n_host;
		i_bottom = (i - 1 + n_host) % n_host;
		for(j = 0; j < n_host; j += 1) {
			j_right = (j + 1) % n_host;
			j_left = (j - 1 + n_host) % n_host;
			field_host_new[i * n_host + j] = (1.0 - 4.0 * theta_host) * field_host[i * n_host + j]
				+ theta_host * (field_host[i_top * n_host + j] + field_host[i_bottom * n_host + j]
					      + field_host[i * n_host + j_right] + field_host[i * n_host + j_left]);

		}
	}
}

int main(void) {
//delcare variavles-------------------------------------------------------------
	int i;
	int j;
	int k;
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
	char filename_write[256];

//initialize--------------------------------------------------------------------
	//set variables---------------------------------------------------------
	n_host = N;
	n_square = N * N;
	l_host = L;
	theta_host = THETA;
	dim_threads.x = NT;
	dim_threads.y = NT;
	dim_threads.z = 1;
	n_blocks = (int)(ceil((double)n_host / NT));
	iteration = M;

	//allocate memories-----------------------------------------------------
	cudaMemcpyToSymbol(n, &n_host, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(theta, &theta_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaHostAlloc((void **)&field_host[0], n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&field_host[1], n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&result_global_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&result_shared_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaMalloc((void **)&field_device[0], n_square * sizeof(double));
	cudaMalloc((void **)&field_device[1], n_square * sizeof(double));

	//calculate using only global memory------------------------------------
		//initialize field----------------------------------------------
	init_field(field_host[0], n_host, l_host);
	cudaMemcpy(field_device[0], field_host[0], n_square * sizeof(double), cudaMemcpyHostToDevice);
		//iteration-----------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_global<<<n_blocks, dim_threads>>>(field_device[i], field_device[j]);
		cudaDeviceSynchronize();
		flip_ij(&i, &j);
	}
		//copy to host and print out------------------------------------
	cudaMemcpy(field_host[0], field_device[i], n_square * sizeof(double), cudaMemcpyDeviceToHost);
	sprintf(filename_write, "result_global.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, field_host[0], n_host, l_host);
	fclose(file_write);

	//calculate on CPU------------------------------------------------------
		//initialize field----------------------------------------------
	init_field(field_host[0], n_host, l_host);
		//iteration-----------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_host(field_host[i], field_host[j], n_host, theta_host);
		flip_ij(&i, &j);
	}
		//print out-----------------------------------------------------
	sprintf(filename_write, "result_host.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, field_host[i], n_host, l_host);
	fclose(file_write);



//finalize----------------------------------------------------------------------
	cudaFreeHost(field_host[0]);
	cudaFreeHost(field_host[1]);
	cudaFreeHost(result_global_host);
	cudaFreeHost(result_shared_host);
	cudaFree(field_device[0]);
	cudaFree(field_device[1]);

	return 0;
}
