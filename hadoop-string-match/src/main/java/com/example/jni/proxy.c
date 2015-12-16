#include <stdio.h>
#include "com_example_jni_CudaWrapper.h"

JNIEXPORT jint JNICALL Java_com_example_jni_CudaWrapper_CUDAProxy_1stringMatch(JNIEnv *env, jobject obj, jbyteArray word_array, jstring matching_word, jint device_id)
{
	int count = 0;
	jsize size = 0;
	printf("C: fetching arrays from Java\n");

	jbyte *words = (*env)->GetByteArrayElements(env, word_array, 0);
	//const char *words = (*env)->GetStringUTFChars(env, word_array, 0);
	const char *matchingword = (*env)->GetStringUTFChars(env, matching_word, 0);

	count = cuda_string_match(words, matchingword, device_id);
	
	(*env)->ReleaseByteArrayElements(env, word_array, words, 0);
	//(*env)->ReleaseStringUTFChars(env, word_array, words);
	(*env)->ReleaseStringUTFChars(env, matching_word, matchingword);

	return (jint)count; // this might not be the right way to return values to Java
}
