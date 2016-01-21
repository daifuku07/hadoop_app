for i in `seq 1 1`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	hadoop fs -rm -r cpu_output
	hadoop fs -rm -r cpu_input
	hadoop fs -put input3 input
	hadoop fs -put ../hadoop-matrix-calculation-cpu/input_512 cpu_input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native

	cd ../hadoop-matrix-calculation-cpu
	sh run_script.sh &
	cd ../hadoop-memory-copy
	hadoop jar hadoop-examples-gpu-memcopy.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha -D mapred.child.java.opts=-Xmx1024m -D yarn.app.mapreduce.am.resource.mb=1024 -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=512 input output GPU &
	wait 
	exit
done
