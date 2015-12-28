for i in `seq 1 1`
do
	#hadoop fs -rm -r native
	hadoop fs -rm -r cpu_input
	hadoop fs -rm -r cpu_output
	hadoop fs -put input_512 cpu_input
	#hadoop fs -put input_scheduling cpu_input
	#hadoop fs -mkdir native
	#hadoop fs -put src/main/java/com/example/jni/program.so native
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc input output CPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.child.java.opts=-Xmx512m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=1024  input output GPU
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=default -D mapred.child.java.opts=-Xmx1024m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=0 cpu_input cpu_output CPU
done
