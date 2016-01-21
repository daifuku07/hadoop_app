#include <stdio.h>
//#include <stdlib.h>
#include <unistd.h>

#define BLOCK_SIZE 32
//#define GRID_SIZE 1920
#define GRID_SIZE 2200
//#define GRID_SIZE 3000

extern "C" {

	__global__ void mul_matrix(int *A, int size){
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;
		int num = x + GRID_SIZE * y;

		if(num < size){
			A[num] = num+1;
		}
		else{
			A[0] += 1;
		}
	}

	// CUDA code here
	int cuda_matrixMul(int *a_h, int size, int device_id){
		cudaError_t err;
		int *a_d;

		printf("C: device id >> %d\n", device_id);
		cudaSetDevice(device_id);

		//printf("C: Allocate GPU Memory1\n");
		// allocate memory in the GPU device for a, b and c
		err = cudaMalloc((void **) & a_d, size);
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

		int N = size / 4;
		// do calculations on device
		dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
		//dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
		//dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
		//dim3 grid(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(GRID_SIZE, GRID_SIZE, 1);

		// Launch GPU
		printf("C: Launch(size = %d)\n", N);
		mul_matrix<<<grid, block>>>(a_d, N);
		cudaDeviceSynchronize();

		//printf("C: Memory Copy device to host\n");
		err = cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			printf("CUDA error(4): %s\n", cudaGetErrorString(err));
			exit(-1);
		}

		printf("C: Mismatch (a[%d] = %d)\n", 0 , a_h[0]);
		
		for(int i = 1; i < (size/4); i++) {
			if(a_h[i] != i+1){
				printf("C: Mismatch (a[%d] = %d)\n", i , a_h[i]);
				exit(-1);
			}
		}

		//printf("C: Memory Free\n");
		cudaFree(a_d);

		return 0;
	}
}
