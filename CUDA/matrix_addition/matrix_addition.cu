#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <driver_types.h>

void matrixAdd(float *,float *,float *,int,int);
__global__ void matrixAddKernel(float *,float *,float *,int);

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
			B[i*ncol+j] = ((float)rand())/RAND_MAX;
			C[i*ncol+j] = ((float)rand())/RAND_MAX;
			printf("B: %f - C: %f\n", B[i*ncol+j],C[i*ncol+j]);
		}
	}


	matrixAdd(A,B,C,nrow,ncol);
	FILE *output = fopen("matrix_output.txt", "w");
	if(output == NULL){
		printf("A file wasn't created or located\n");
		exit(EXIT_FAILURE);
	}

	for(i=0;i<nrow;i++)
	{
		for(j=0;j<ncol;j++)
		{
			fprintf(output,"%f ",A[i*ncol+j]);
		}
		fprintf(output,"\n");
	}


	free(A);
	free(B);
	free(C);
	return 0;
}

void matrixAdd(float * h_A,float * h_B, float * h_C, int nrow,int ncol){
	int size = nrow * ncol * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaError_t error = cudaMalloc((void **)&d_B, size);
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
	cudaMemcpy(d_C,h_C,size,cudaMemcpyHostToDevice);

	error = cudaMalloc((void **)&d_A, size);
	if(error != cudaSuccess){
		printf("%s in %s at line %d \n", cudaGetErrorString(error), __FILE__   ,__LINE__);
		exit(EXIT_FAILURE);
	}

	//run kernel function with 32 threads for each block
	matrixAddKernel<<<ceil(ncol/32.0), 32>>>(d_A,d_B,d_C,ncol);


	cudaMemcpy(h_A,d_A,size,cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

__global__
void matrixAddKernel(float * A,float * B, float * C, int n){
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	if(j < n){
		int i;
		for(i=0;i<n;i++){
			A[j+n*i] = B[j+n*i] + C[j+n*i];
		}
	}

}
