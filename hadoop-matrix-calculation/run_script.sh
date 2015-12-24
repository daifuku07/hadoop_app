for i in `seq 1 1`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	#hadoop fs -put input_1024 input
	hadoop fs -put input_scheduling input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha input output GPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.child.java.opts=-Xmx512m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=1024  input output GPU
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha -D mapred.child.java.opts=-Xmx512m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=1024  input output GPU
done
