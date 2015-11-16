#include <stdio.h>

int main(int argc, char * argv[]){
	int i = 0, j = 0;
	int size = 0;
	
	size = atoi(argv[1]);
	
	printf("%d\n", size);
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			printf("%d ", i);
		}
	}


	return 0;
}
