for i in `seq 1 1`
do
	hadoop fs -rm -r matrix_native
	hadoop fs -rm -r matrix_input
	hadoop fs -rm -r /user/master/matrix_output
	#hadoop fs -put input_2048 matrix_input
	#hadoop fs -put input_2560 matrix_input
	#hadoop fs -put input_3072 matrix_input
	hadoop fs -put input_4096 matrix_input
	hadoop fs -mkdir matrix_native
	hadoop fs -put src/main/java/com/example/jni/program.so matrix_native
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha input output GPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.child.java.opts=-Xmx512m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=1024  input output GPU
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha -D mapred.child.java.opts=-Xmx2048m -D mapreduce.map.memory.mb=2048 -D mapreduce.map.gpu-memory.mb=256  matrix_input matrix_output GPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha -D mapred.child.java.opts=-Xmx1024m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=128  matrix_input matrix_output GPU
done
