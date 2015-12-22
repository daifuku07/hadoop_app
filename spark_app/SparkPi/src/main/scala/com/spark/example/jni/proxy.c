#include <stdio.h>
#include "com_spark_example_jni_CudaWrapper.h"

JNIEXPORT jint JNICALL Java_com_spark_example_jni_CudaWrapper_CUDAProxy_1matrixMul(JNIEnv *env, jobject obj, jintArray aArray, jintArray bArray, jintArray cArray, jint n, jint device_id)
{
	int i = 0, j = 0;
	jsize N = 0;
	//printf("C: fetching arrays from Java\n");
	
	jint *a = (*env)->GetIntArrayElements(env, aArray, 0);
	jint *b = (*env)->GetIntArrayElements(env, bArray, 0);
	jint *c = (*env)->GetIntArrayElements(env, cArray, 0);
		
	//printf("C: Got reference to all a, b, and c\n");
	N = (*env)->GetArrayLength(env, cArray);
	//printf("C: calling CUDA kernel\n");
	//printf("C: array size = %d\n", N);
	
	cuda_matrixMul(a, b, c, n, device_id);
	
	//printf("C: back from CUDA kernel, coping data to Java\n");
	(*env)->ReleaseIntArrayElements(env, aArray, a, 0);
	(*env)->ReleaseIntArrayElements(env, bArray, b, 0);
	(*env)->ReleaseIntArrayElements(env, cArray, c, 0);
	//printf("C: Going back to Java\n");

	return (jint) N; // this might not be the right way to return values to Java
}


JNIEXPORT jint JNICALL Java_com_spark_example_jni_CudaWrapper_CUDAProxy_1printHello(JNIEnv *env, jobject obj, jint num)
{

	printf("Native C: Hello: %d. \n", num);
}	
