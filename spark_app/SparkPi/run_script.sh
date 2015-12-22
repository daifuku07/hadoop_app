for i in `seq 1 1`
do
	hadoop fs -rm -r native
	hadoop fs -rm -r input
	hadoop fs -rm -r output
	hadoop fs -mkdir native
	hadoop fs -put src/main/scala/com/spark/example/jni/program.so native
	spark-submit --class com.spark.example.SparkPi --master yarn-client --executor-memory 1G --executor-gpu-memory 256MB target/SparkPi-1.4.1.jar 2
done
