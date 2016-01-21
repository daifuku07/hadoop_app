#include <stdio.h>
//#include <stdlib.h>
#include <unistd.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 38400

extern "C" {
	__global__ void mul_matrix(int *A, int *B, int *C, int size){
		int i = threadIdx.x + blockDim.x * blockIdx.x;
		int sum = 0;

		if(i < size){
			__syncthreads();
			sum = A[i] + B[i];
			C[i] = sum;
		}
	}

	// CUDA code here
	int cuda_matrixMul(int *a_h, int *b_h, int *c_h, int size, int device_id){
		cudaError_t err;
		int *a_d, *b_d, *c_d;

		printf("C: device id >> %d\n", device_id);
		cudaSetDevice(device_id);

		//printf("C: Allocate GPU Memory1\n");
		// allocate memory in the GPU device for a, b and c
		err = cudaMalloc((void **) & a_d, size);
		if (err != cudaSuccess){
			printf("CUDA error(1): %s\n", cudaGetErrorString(err));
			exit(-1);
		}
		err = cudaMalloc((void **) & b_d, size);
		if (err != cudaSuccess){
			printf("CUDA error(1): %s\n", cudaGetErrorString(err));
			exit(-1);
		}
		err = cudaMalloc((void **) & c_d, size);
		if (err != cudaSuccess){
			printf("CUDA error(1): %s\n", cudaGetErrorString(err));
			exit(-1);
		}

		// copy from host to GPU device
		//printf("C: Memory Copy tost to device(Size %d)\n", size);
		err = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
			printf("CUDA error(4): %s\n", cudaGetErrorString(err));
			exit(-1);
		}
		err = cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
			printf("CUDA error(4): %s\n", cudaGetErrorString(err));
			exit(-1);
		}

		int N = size / 4;
		// do calculations on device
		dim3 block(BLOCK_SIZE, 1, 1);
		dim3 grid(N/BLOCK_SIZE, 1, 1);

		// Launch GPU
		printf("C: Launch(size = %d)\n", N);
		mul_matrix<<<grid, block>>>(a_d, b_d, c_d, N);
		cudaDeviceSynchronize();

		//printf("C: Memory Copy device to host\n");
		err = cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			printf("CUDA error(4): %s\n", cudaGetErrorString(err));
			exit(-1);
		}
		cudaDeviceSynchronize();

		for(int i = 0; i < (size/4); i++) {
			if(c_h[i] != i+i){
				printf("C: Mismatch (c[%d] = %d)\n", i , c_h[i]);
				exit(-1);
			}
		}

		//printf("C: Memory Free\n");
		cudaFree(a_d);

		return 0;
	}
}
