for i in `seq 1 1`
do
	#hadoop fs -rm -r cpu_input
	#hadoop fs -rm -r cpu_output
	#hadoop fs -put input_512 cpu_input
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc input output CPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.child.java.opts=-Xmx512m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=1024  input output GPU
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=beta -D mapred.child.java.opts=-Xmx256m -D yarn.app.mapreduce.am.resource.mb=1024 -D mapreduce.map.memory.mb=512 -D mapreduce.map.gpu-memory.mb=0 cpu_input cpu_output CPU >> logs/hadoop_extended_sche_10_log.txt 2>&1
done
