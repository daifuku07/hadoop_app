for i in `seq 1 1`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	hadoop fs -rm -r output2
	hadoop fs -put input3 input
	#hadoop fs -put input_scheduling input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native
	#cd ../hadoop-memory-copy2
	#sh run_script.sh &
	#cd ../hadoop-memory-copy
	hadoop jar hadoop-examples-gpu-matrix-add.jar com.example.hadoop.MatrixCalc -D mapred.job.queue.name=alpha -D mapred.child.java.opts=-Xmx1024m -D yarn.app.mapreduce.am.resource.mb=512 -D mapreduce.map.memory.mb=1024 -D mapreduce.map.gpu-memory.mb=512 input output GPU 
	#wait 
	#exit
done
