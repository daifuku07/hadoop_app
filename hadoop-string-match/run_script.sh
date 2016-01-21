for i in `seq 1 1`
do
	hadoop fs -rm -r string_native
	hadoop fs -rm -r string_input
	hadoop fs -rm -r string_output
	#hadoop fs -put input_20mb input
	#hadoop fs -put input_40mb string_input
	hadoop fs -put input_400mb string_input
	#hadoop fs -put input_160mb input
	#hadoop fs -put input_100mb input
	#hadoop fs -put input_200mb string_input
	#hadoop fs -put input_150mb string_input
	#hadoop fs -put input_640mb input
	hadoop fs -mkdir string_native
	hadoop fs -put src/main/java/com/example/jni/program.so string_native
	#hadoop jar hadoop-examples-string-match.jar com.example.hadoop.StringMatch -D mapred.job.queue.name=beta -D mapred.child.java.opts=-Xmx1024m -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=256 string_input string_output GPU
	hadoop jar hadoop-examples-string-match.jar com.example.hadoop.StringMatch -D mapred.job.queue.name=beta -D mapred.child.java.opts=-Xmx2048m -D mapreduce.map.memory.mb=2048 -D mapreduce.map.gpu-memory.mb=512 string_input string_output GPU
done
