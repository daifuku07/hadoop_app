package com.example.jni;
/**
 * Created by gpu on 15/11/09.
 */
public class CudaWrapper {
  public CudaWrapper(String path){
    System.load(path);
  }

  public native int CUDAProxy_matrixMul(float[] a, float[] b, float[] c, int n, int device_id);
}
