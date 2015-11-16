#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <driver_types.h>

#define BLOCK_SIZE 32

void matrixMul(float *,float *,float *,int,int);
__global__ void matrixMulKernel(float *,float *,float *,int);

int main(int argc, char * argv[]){
	int i,j,nrow,ncol;//,rows,columns,i;
	float *A, *B, *C;

	nrow = atoi(argv[1]);
	ncol = atoi(argv[2]);

	if((nrow != ncol) || argc != 3){
		printf("Number of rows should be equal to number of columns\n");
		exit(EXIT_FAILURE);
	}
	int size = nrow * ncol * sizeof(float);
	A = (float *)malloc(size);
	B = (float *)malloc(size);
	C = (float *)malloc(size);

	srand(time(NULL));
	for(i=0;i<nrow;i++)
	{
		for(j=0;j<ncol;j++)
		{
			A[i*ncol+j] = (float)i;
			B[i*ncol+j] = (float)i;
			//B[i*ncol+j] = ((float)rand())/RAND_MAX;
			//C[i*ncol+j] = ((float)rand())/RAND_MAX;
			printf("%1.0f ", B[i*ncol+j]);
		}
		printf("\n");
	}


	matrixMul(A,B,C,nrow,ncol);
	FILE *output = fopen("matrix_output.txt", "w");
	if(output == NULL){
		printf("A file wasn't created or located\n");
		exit(EXIT_FAILURE);
	}

	for(i=0;i<nrow;i++)
	{
		for(j=0;j<ncol;j++)
		{
			printf("%1.0f ", C[i*ncol+j]);
			//fprintf(output,"%1f ", A[i*ncol+j]);
		}
		printf("\n");
		//fprintf(output,"\n");
	}

	printf("End\n");

	free(A);
	free(B);
	free(C);
	return 0;
}

void matrixMul(float * h_A,float * h_B, float * h_C, int nrow,int ncol){
	int size = nrow * ncol * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaError_t error = cudaMalloc((void **)&d_A, size);
	if(error != cudaSuccess){
		printf("%s in %s at line %d \n", cudaGetErrorString(error), __FILE__   ,__LINE__);
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);

	error = cudaMalloc((void **)&d_B, size);
	if(error != cudaSuccess){
		printf("%s in %s at line %d \n", cudaGetErrorString(error), __FILE__   ,__LINE__);
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

	error = cudaMalloc((void **)&d_C, size);
	if(error != cudaSuccess){
		printf("%s in %s at line %d \n", cudaGetErrorString(error), __FILE__   ,__LINE__);
		exit(EXIT_FAILURE);
	}

	//run kernel function with 32 threads for each block
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	//dim3 grid((d_C.width+block.x - 1) / block.x, (d_C.height+block.y - 1) / block.y);
	matrixMulKernel<<<grid, block>>>(d_A,d_B,d_C,ncol);


	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

__global__
void matrixMulKernel(float * A,float * B, float * C, int n){
	unsigned int i;
	float product = 0;
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < n && col < n){
		for (i = 0; i < n; i++)
			product += A[row * n + i] * B[i * n + col];

		C[row*n + col] = (float)product;
	}
}
