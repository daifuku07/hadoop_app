package com.spark.example

import com.spark.example.jni.CudaWrapper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext, TaskContext}
import scala.math._


/** Computes an approximation to pi */
object SparkPi {
   def main(args: Array[String]) {
     val conf = new SparkConf().setAppName("Spark Pi")
     val spark = new SparkContext(conf)
     val slices = if (args.length > 0) args(0).toInt else 2
     val n = math.min(100000L * slices, Int.MaxValue).toInt // avoid overflow

     val m = new CudaWrapper("/home/gpu/workspace/hadoop_app/spark_app/SparkPi/src/main/scala/com/spark/example/jni/program.so")
     m.CUDAProxy_printHello(5)

     val count = spark.parallelize(1 until n, slices).map { i =>
         val x = random * 2 - 1
         val y = random * 2 - 1
         println("GPU Device ID: " + TaskContext.get().gpuDeviceId())
         if (x*x + y*y < 1) 1 else 0
       }.reduce(_ + _)
     println("Pi is roughly " + 4.0 * count / n)
     spark.stop()
   }
 }
