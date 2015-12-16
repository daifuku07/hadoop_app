#include <stdio.h>
#include <stdlib.h>
#include <jni.h>

#define BLOCK_SIZE 384

//#define DEBUG

extern "C" {

	__global__ void string_match(char *words, char *matching_word, int *count, int size, int msize, int block_size){
		int offset = size / block_size;
		int init_flag = 0;
		int finish_flag = 0;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    count[col] = 0;

		int startpoint = col * offset;
		int endpoint = col * offset + offset;
		int i = 0, k = 0;

		for(i = startpoint; finish_flag == 0; i++){
			if(words[i] == ',' || words[i] == '.' || words[i] == '\n' || words[i] == ' ' || words[i] == '\0'){
				if(k == msize){
					count[col]++;
				}
				k = 0;
				init_flag = 1;

				if(i >=  endpoint){
					finish_flag = 1;
				}
			}
			if(k > msize && init_flag == 1){
				init_flag = 0;
			}
			if(init_flag == 1){
				if(words[i] == matching_word[k]){
					k++;
				}
			}
		}
	}

	// CUDA code here
	int cuda_string_match(jbyte *words, char *matching_word, int device_id){
		cudaError_t err;

		char *words_d, *matching_word_d;
		char *words_h = (char *)words;
		int *count_d;
		int *count;
		count = (int *)malloc(BLOCK_SIZE);
		int word_num = strlen(words_h);
		size_t words_size = word_num * sizeof(char);

		int i = 0, k = 0;

		//printf("size = %d, msize = %d\n", word_num, (int)strlen(matching_word));

#ifdef DEBUG
		printf("Input words: \n");
		for(i = 0; i < BLOCK_SIZE; i++){
			printf("Input words(%d): \n", i);
			for(k = 0; k < word_num/BLOCK_SIZE; k++){
						printf("%c", words[(word_num/BLOCK_SIZE) * i + k]);
			}
			printf("\n");
		}
		printf("End words: ");
 #endif

		printf("C: device id >> %d\n", device_id);
		cudaSetDevice(device_id);

		printf("C: matching_word = %s\n", matching_word);

		// allocate memory in the GPU device
    err = cudaMalloc((void **) &words_d, words_size);
    if (err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
    	exit(-1);
    }
    err = cudaMalloc((void **) &matching_word_d, strlen(matching_word)*sizeof(char));
    if (err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
    	exit(-1);
    }
		err = cudaMalloc((void **) &count_d, BLOCK_SIZE*sizeof(int));
    if (err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
    	exit(-1);
    }

		// copy from host to GPU device
		err = cudaMemcpy(words_d, words, words_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){
			printf("CUDA error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    err = cudaMemcpy(matching_word_d, matching_word, strlen(matching_word)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
      exit(-1);
     }

		// do calculations on device
		//dim3 block(BLOCK_SIZE, 1);
		//dim3 grid(BLOCK_SIZE, 1);

		// Launch GPU
		string_match<<<1, BLOCK_SIZE>>>(words_d, matching_word_d, count_d, word_num, strlen(matching_word), BLOCK_SIZE);

		cudaMemcpy(count, count_d, BLOCK_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(words_d);
		cudaFree(matching_word_d);
		cudaFree(count_d);

		int ans_count = 0;

		//printf("Output count: ");
    for(i = 0; i < BLOCK_SIZE; i++){
    	ans_count += count[i];
    	//printf("%d, ", count[i]);
    }
    //printf("End Output: ans_count = %d\n", ans_count);

		return ans_count;
	}
}
