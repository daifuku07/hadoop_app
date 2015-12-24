#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h>

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
		cudaError_t err;
		int *a_d, *b_d, *c_d;
		//size_t size = N * N * sizeof (int);
		// Test 300MB
		//size_t size = 314572800;
		// Test 156MB
		size_t size = 163577856;

		printf("C: device id >> %d\n", device_id);
		cudaSetDevice(device_id);

		printf("C: Allocate GPU Memory1\n");
		// allocate memory in the GPU device for a, b and c
		err = cudaMalloc((void **) & a_d, size);
		if (err != cudaSuccess){
        	printf("CUDA error(1): %s\n", cudaGetErrorString(err));
        	exit(-1);
        }
    printf("C: Allocate GPU Memory2\n");
		err = cudaMalloc((void **) & b_d, size);
		if (err != cudaSuccess){
        	printf("CUDA error(2): %s\n", cudaGetErrorString(err));
        	exit(-1);
        }
    printf("C: Allocate GPU Memory3\n");
		err = cudaMalloc((void **) & c_d, size);
		if (err != cudaSuccess){
        	printf("CUDA error(3): %s\n", cudaGetErrorString(err));
        	exit(-1);
        }

		size = N * N * sizeof (int);

		// copy from host to GPU device
		printf("C: Memory Copy1\n");
		err = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
            	printf("CUDA error(4): %s\n", cudaGetErrorString(err));
            	exit(-1);
    }
    printf("C: Memory Copy2\n");
		err = cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
                	printf("CUDA error(5): %s\n", cudaGetErrorString(err));
                	exit(-1);
    }


		// do calculations on device
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

		// Launch GPU
		printf("C: Launch\n");
		mul_matrix<<<grid, block>>>(a_d, b_d, c_d, N);

		printf("C: Memory Copy\n");
		cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
		
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);

		return N;
	}
}
