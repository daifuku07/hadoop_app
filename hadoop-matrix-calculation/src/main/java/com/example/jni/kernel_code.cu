#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 32

extern "C" {

	__global__ void add_matrix(float *A, float *B, float *C, int n){
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

	// CUDA code here
	int cuda_matrixAdd(float *a_h, float *b_h, float *c_h, int N){
		float *a_d, *b_d, *c_d;
		size_t size = N * N * sizeof (float);

		/*
		float *a = (float *)malloc(N * N * sizeof(float));
		float *b = (float *)malloc(N * N * sizeof(float));

		for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				a[i] = i;
				b[i] = i;
			}
		}

		a_h = a;
		b_h = b;
		*/

		// allocate memory in the GPU device for a, b and c
		cudaMalloc((void **) & a_d, size);
		cudaMalloc((void **) & b_d, size);
		cudaMalloc((void **) & c_d, size);

		/*
		int s = 0;
		while(1){
			sleep(3);
			s++;
			sleep(3);
			s--;
			sleep(3);
		}
	*/

		// copy from host to GPU device
		cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

		// do calculations on device
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

		// Launch GPU
		add_matrix <<<grid, block >>>(a_d, b_d, c_d, N);

		cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
		
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);

		return N;
	}

}
