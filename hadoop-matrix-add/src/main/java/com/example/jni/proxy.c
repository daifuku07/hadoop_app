#include <stdio.h>
#include "com_example_jni_CudaWrapper.h"

JNIEXPORT jint JNICALL Java_com_example_jni_CudaWrapper_CUDAProxy_1matrixMul(JNIEnv *env, jobject obj,
jintArray aArray, jintArray bArray, jintArray cArray, jint size, jint device_id){
	jint retVal = 0;
	//printf("C: fetching arrays from Java\n");
	
	jint *a = (*env)->GetIntArrayElements(env, aArray, 0);
	jint *b = (*env)->GetIntArrayElements(env, bArray, 0);
	jint *c = (*env)->GetIntArrayElements(env, cArray, 0);

	retVal = cuda_matrixMul(a, b, c, size, device_id);
	
	//printf("C: back from CUDA kernel, coping data to Java\n");
	(*env)->ReleaseIntArrayElements(env, aArray, a, 0);
	(*env)->ReleaseIntArrayElements(env, bArray, b, 0);
	(*env)->ReleaseIntArrayElements(env, cArray, c, 0);

	return retVal; // this might not be the right way to return values to Java
}
