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



