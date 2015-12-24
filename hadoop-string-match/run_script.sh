for i in `seq 1 5`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	#hadoop fs -put input_20mb input
	#hadoop fs -put input_40mb input
	#hadoop fs -put input_80mb input
	hadoop fs -put input_100mb input
	#hadoop fs -put input_160mb input
	#hadoop fs -put input_640mb input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.StringMatch input output CPU
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.StringMatch input output GPU
done
