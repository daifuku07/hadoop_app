for i in `seq 1 5`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	hadoop fs -put input_512 input
	hadoop fs -mkdir native
	hadoop fs -put src/main/java/com/example/jni/program.so native
	#hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc input output CPU
	hadoop jar hadoop-examples-matrix-calculation.jar com.example.hadoop.MatrixCalc input output GPU
done
