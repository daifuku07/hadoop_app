#include <stdio.h>

const int N = 20; 
const int MAX_WORD_SIZE = 1024;

	__global__ 
void hello(char *a, char *b, int *c, int size, int msize) 
{
	int i = 0;
	
	for(i = 0; i < msize; i++){
		if(a[N * threadIdx.x + i] != b[i]){
			c[threadIdx.x] = 0;
			break;
		}
		if(i == msize - 1){
			c[threadIdx.x] = 1;
			break;
		}
	}
}

int main()
{
	char words[MAX_WORD_SIZE][N];
	FILE *fp;
	char *fname = "input_text.txt";
	int c;

	int i = 0;
	int word_count = 0;

	// File Load
	fp = fopen( fname, "r" );
	if( fp == NULL ){
		printf( "%sファイルが開けません¥n", fname );
		return -1;
	}

	while( (c = fgetc( fp )) != EOF ){
		if(c != ',' && c != '.' && c != ' ' && c != '\n' && c != '\t' && c != '\0'){
			//printf("%c", c);
			words[word_count][i] = c;
			i++;
		}
		else{
			if(i != 0){
				//printf("\n");
				words[word_count][i] = '\0';
				word_count++;
				i = 0;
			}
		}
	}

	/*
	printf("Input: \n");
	for(i = 0; i < word_count; i++){
		printf("%s\n", words[i]);
	}
	*/
	char match_word[N] = {"in"};

	int count[word_count];

	char *ad, *bd;
	int *cd;

	const int csize = word_count*N*sizeof(char);
	const int bsize = strlen(match_word)*sizeof(char);
	const int isize = word_count*sizeof(int);

	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&bd, bsize ); 
	cudaMalloc( (void**)&cd, isize );

	cudaMemcpy( ad, words, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, match_word, bsize, cudaMemcpyHostToDevice ); 

	dim3 dimBlock( word_count, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd, cd, word_count, bsize);
	cudaMemcpy( count, cd, isize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	int ans = 0;

	for(i = 0; i < word_count; i++){
		ans += count[i];
		//printf("%s, %d\n", words[i], count[i]);
	}
	printf("%s: %d\n", match_word, ans);
	
	return EXIT_SUCCESS;
}
