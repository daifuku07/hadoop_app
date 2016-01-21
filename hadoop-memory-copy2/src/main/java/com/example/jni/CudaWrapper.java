package com.example.jni;
/**
 * Created by gpu on 15/11/09.
 */
public class CudaWrapper {
  public CudaWrapper(String path){
    System.load(path);
  }

  public native int CUDAProxy_matrixMul(int[] a, int size, int device_id);
}
