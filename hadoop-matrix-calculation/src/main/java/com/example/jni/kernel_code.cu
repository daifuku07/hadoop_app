#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

extern "C" {

	__global__ void mul_matrix(int *A, int *B, int *C, int n){
		unsigned int i;
		int product = 0;

		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		if(row < n && col < n){
			for (i = 0; i < n; i++)
				product += A[row * n + i] * B[i * n + col];

			C[row*n + col] = product;
		}
	}

	// CUDA code here
	int cuda_matrixMul(int *a_h, int *b_h, int *c_h, int N, int device_id){
		int *a_d, *b_d, *c_d;
		size_t size = N * N * sizeof (int);

		printf("C: device id >> %d\n", device_id);

		// allocate memory in the GPU device for a, b and c
		cudaMalloc((void **) & a_d, size);
		cudaMalloc((void **) & b_d, size);
		cudaMalloc((void **) & c_d, size);


		// copy from host to GPU device
		cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

		// do calculations on device
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

		// Launch GPU
		mul_matrix<<<grid, block>>>(a_d, b_d, c_d, N);

		cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
		
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);

		return N;
	}
}
