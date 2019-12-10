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
	int i_left, i_right;
	int j_top, j_bottom;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	if(i_global < n) {
		i_right = (i_global + 1) % n;
		i_left = (i_global - 1 + n) % n;
		for(j_global = threadIdx.y; j_global < n; j_global += NT) {
			j_top = (j_global + 1) % n;
			j_bottom = (j_global - 1 + n) % n;
			field_device_new[i_global * n + j_global] = (1.0 - 4.0 * theta) * field_device[i_global * n + j_global]
				+ theta * (field_device[i_right * n + j_global] + field_device[i_left * n + j_global]
					      + field_device[i_global * n + j_top] + field_device[i_global * n + j_bottom]);
		}
	}
}

__global__ void diffusion_global_transpose(double *field_device, double *field_device_new) {
	int i_global;
	int j_global;
	int i_left, i_right;
	int j_top, j_bottom;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	if(i_global < n) {
		i_right = (i_global + 1) % n;
		i_left = (i_global - 1 + n) % n;
		for(j_global = threadIdx.y; j_global < n; j_global += NT) {
			j_top = (j_global + 1) % n;
			j_bottom = (j_global - 1 + n) % n;
			field_device_new[i_global + j_global * n] = (1.0 - 4.0 * theta) * field_device[i_global + j_global * n]
				+ theta * (field_device[i_right + j_global * n] + field_device[i_left + j_global * n]
					      + field_device[i_global + j_top * n] + field_device[i_global + j_bottom * n]);
		}
	}
}

__global__ void diffusion_shared(double *field_device, double *field_device_new) {
	int i_global;
	int j_global;
	int i_shared;
	int j_shared;
	int i_left, i_right;
	int j_top, j_bottom;
	double field_register;
	__shared__ double field_shared[(NT + 2) * (NT + 2)];

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	i_shared = threadIdx.y + 1;
	j_shared = threadIdx.x + 1;

	if(i_global < n) {
		i_right = (i_global + 1) % n;
		i_left = (i_global - 1 + n) % n;
		for(j_global = threadIdx.y; j_global < n; j_global += NT) {
			j_top = (j_global + 1) % n;
			j_bottom = (j_global - 1 + n) % n;
			//copy field from global to shared----------------------
			field_register = field_device[i_global * n + j_global];
			field_shared[i_shared * (NT + 2) + j_shared] = field_register;
			if(i_shared == 1) {
				field_shared[0 * (NT + 2) + j_shared] = field_device[i_left * n + j_global];
			} else if(i_shared == NT) {
				field_shared[(NT + 1) * (NT + 2) + j_shared] = field_device[i_right * n + j_global];
			} else if(j_shared == 1) {
				field_shared[i_shared * (NT + 2) + 0] = field_device[i_global * n + j_bottom];
			} else if(j_shared == NT) {
				field_shared[i_shared * (NT + 2) + NT + 1] = field_device[i_global * n + j_top];
			}
			__syncthreads();
			//calculate field evolution-----------------------------
			field_device_new[i_global * n + j_global] = (1.0 - 4.0 * theta) * field_register
				+ theta * (field_shared[i_right * n + j_shared] + field_shared[i_left * n + j_shared]
					      + field_shared[i_shared * n + j_top] + field_shared[i_shared * n + j_bottom]);

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
		y = (double)j * d;
		for(j = 0; j < N; j += 1) {
			x = (double)i * d;
			fprintf(file_write, "%f %f %f\n", x, y, field[i * n + j]);
		}
	}
}

void diffusion_host(double *field_host, double *field_host_new, int n_host, double theta_host) {
	int i;
	int j;
	int i_right, i_left;
	int j_top, j_bottom;

	for(i = 0; i < n_host; i += 1) {
		i_right = (i + 1) % n_host;
		i_left = (i - 1 + n_host) % n_host;
		for(j = 0; j < n_host; j += 1) {
			j_top = (j + 1) % n_host;
			j_bottom = (j - 1 + n_host) % n_host;
			field_host_new[i * n_host + j] = (1.0 - 4.0 * theta_host) * field_host[i * n_host + j]
				+ theta_host * (field_host[i_right * n_host + j] + field_host[i_left * n_host + j]
					      + field_host[i * n_host + j_top] + field_host[i * n_host + j_bottom]);

		}
	}
}

void diffusion_host_transpose(double *field_host, double *field_host_new, int n_host, double theta_host) {
	int i;
	int j;
	int i_right, i_left;
	int j_top, j_bottom;

	for(i = 0; i < n_host; i += 1) {
		i_right = (i + 1) % n_host;
		i_left = (i - 1 + n_host) % n_host;
		for(j = 0; j < n_host; j += 1) {
			j_top = (j + 1) % n_host;
			j_bottom = (j - 1 + n_host) % n_host;
			field_host_new[i + n_host * j] = (1.0 - 4.0 * theta_host) * field_host[i + n_host * j]
				+ theta_host * (field_host[i_right + n_host * j] + field_host[i_left + n_host * j]
					      + field_host[i + n_host * j_top] + field_host[i + n_host * j_bottom]);

		}
	}
}


double check_residue(double *field_host, double *field_device, int n_host) {
	int i;
	double residue = 0.0;

	for(i = 0; i < n_host * n_host; i += 1) {
		residue += (field_host[i] - field_device[i]) * (field_host[i] - field_device[i]);;
	}
	return residue;
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
	double *result_host;
	double *result_host_transpose;
	double *result_global_host;
	double *result_global_transpose_host;
	double *result_shared_host;
	FILE *file_write;
	char filename_write[256];
	clock_t start, end;

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
	cudaHostAlloc((void **)&result_global_transpose_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&result_shared_host,  n_square * sizeof(double), cudaHostAllocMapped);
	cudaMalloc((void **)&field_device[0], n_square * sizeof(double));
	cudaMalloc((void **)&field_device[1], n_square * sizeof(double));
	result_host = (double *)malloc(n_square * sizeof(double));
	result_host_transpose = (double *)malloc(n_square * sizeof(double));

//calculate on CPU--------------------------------------------------------------
	//initialize field------------------------------------------------------
	init_field(field_host[0], n_host, l_host);
	start = clock();
	//iteration-------------------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_host(field_host[i], field_host[j], n_host, theta_host);
		flip_ij(&i, &j);
	}
	//save and print out----------------------------------------------------
	memcpy(result_host, field_host[i], n_square * sizeof(double));
	end = clock();
	printf("host:%ld\n", end - start);
	sprintf(filename_write, "result_host.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, result_host, n_host, l_host);
	fclose(file_write);

//calculate on CPU, transposed ver----------------------------------------------
	//initialize field------------------------------------------------------
	init_field(field_host[0], n_host, l_host);
	start = clock();
	//iteration-------------------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_host_transpose(field_host[i], field_host[j], n_host, theta_host);
		flip_ij(&i, &j);
	}
	//save and print out----------------------------------------------------
	memcpy(result_host_transpose, field_host[i], n_square * sizeof(double));
	end = clock();
	printf("host_trans:%ld\n", end - start);
	sprintf(filename_write, "result_host_transpose.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, result_host, n_host, l_host);
	fclose(file_write);

//calculate using only global memory--------------------------------------------
	//initialize field------------------------------------------------------
	init_field(field_host[0], n_host, l_host);
	start = clock();
	cudaMemcpy(field_device[0], field_host[0], n_square * sizeof(double), cudaMemcpyHostToDevice);
	//iteration-------------------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_global<<<n_blocks, dim_threads>>>(field_device[i], field_device[j]);
		cudaDeviceSynchronize();
		flip_ij(&i, &j);
	}
	//copy to host and print out--------------------------------------------
	cudaMemcpy(result_global_host, field_device[i], n_square * sizeof(double), cudaMemcpyDeviceToHost);
	end = clock();
	printf("global:%ld\n", end - start);
	sprintf(filename_write, "result_global.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, result_global_host, n_host, l_host);
	fclose(file_write);

//calculate using only global memory, transposed ver----------------------------
	//initialize field------------------------------------------------------
	init_field(field_host[0], n_host, l_host);
	start = clock();
	cudaMemcpy(field_device[0], field_host[0], n_square * sizeof(double), cudaMemcpyHostToDevice);
	//iteration-------------------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_global_transpose<<<n_blocks, dim_threads>>>(field_device[i], field_device[j]);
		cudaDeviceSynchronize();
		flip_ij(&i, &j);
	}
	//copy to host and print out--------------------------------------------
	cudaMemcpy(result_global_host, field_device[i], n_square * sizeof(double), cudaMemcpyDeviceToHost);
	end = clock();
	printf("global_transpose:%ld\n", end - start);
	sprintf(filename_write, "result_global_transpose.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, result_global_transpose_host, n_host, l_host);
	fclose(file_write);

//calculate using shared memory-------------------------------------------------
	//initialize field------------------------------------------------------
	init_field(field_host[0], n_host, l_host);
	start = clock();
	cudaMemcpy(field_device[0], field_host[0], n_square * sizeof(double), cudaMemcpyHostToDevice);
	//iteration-------------------------------------------------------------
	i = 0;
	j = 1;
	for(k = 0; k < iteration; k += 1) {
		diffusion_shared<<<n_blocks, dim_threads>>>(field_device[i], field_device[j]);
		cudaDeviceSynchronize();
		flip_ij(&i, &j);
	}
	//copy to host and print out--------------------------------------------
	cudaMemcpy(result_shared_host, field_device[i], n_square * sizeof(double), cudaMemcpyDeviceToHost);
	end = clock();
	printf("shared:%ld\n", end - start);
	sprintf(filename_write, "result_shared.txt");
	file_write = fopen(filename_write, "w");
	print_field(file_write, result_shared_host, n_host, l_host);
	fclose(file_write);

//check answers-----------------------------------------------------------------
	printf("global:%f, shared:%f\n", check_residue(result_host, result_global_host, n_host), check_residue(result_host, result_shared_host, n_host) );
	printf("global_trans:%f, host_trans:%f\n", check_residue(result_host, result_global_transpose_host, n_host), check_residue(result_host, result_host_transpose, n_host));

//finalize----------------------------------------------------------------------
	cudaFreeHost(field_host[0]);
	cudaFreeHost(field_host[1]);
	cudaFreeHost(result_global_host);
	cudaFreeHost(result_global_transpose_host);
	cudaFreeHost(result_shared_host);
	cudaFree(field_device[0]);
	cudaFree(field_device[1]);
	free(result_host);
	free(result_host_transpose);

	return 0;
}
