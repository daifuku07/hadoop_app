for i in `seq 1 1`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	#hadoop fs -put input_20mb input
	#hadoop fs -put input_40mb input
	#hadoop fs -put input_80mb input
	hadoop fs -put input_400mb input
	#hadoop fs -put input_160mb input
	#hadoop fs -put input_640mb input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native
	hadoop jar hadoop-examples-string-match.jar com.example.hadoop.StringMatch -D mapred.job.queue.name=default -D mapred.child.java.opts=-Xmx8192m -D mapreduce.map.memory.mb=2049 -D mapreduce.map.gpu-memory.mb=512 input output GPU
done
