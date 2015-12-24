package com.example.jni;
/**
 * Created by gpu on 15/11/09.
 */
public class CudaWrapper {
  public CudaWrapper(String path){
    System.load(path);
  }

  public native int CUDAProxy_matrixMul(int[] a, int[] b, int[] c, int n, int device_id);
}
