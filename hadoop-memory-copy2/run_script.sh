for i in `seq 1 1`
do
	#hadoop fs -rm -r native
	#hadoop fs -rm -r input
	#hadoop fs -rm -r output2
	#hadoop fs -put input input
	#hadoop fs -put input_scheduling input
	#hadoop fs -mkdir native
	#hadoop fs -put src/main/java/com/example/jni/program.so native
	hadoop jar hadoop-examples-gpu-memcopy.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=beta -D mapred.child.java.opts=-Xmx1024m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=512 input output2 GPU >> logs/hadoop_native_sche_10_log.txt 2>&1
	#hadoop jar hadoop-examples-gpu-memcopy.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=beta -D mapred.child.java.opts=-Xmx1024m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=512 input output2 GPU
done
