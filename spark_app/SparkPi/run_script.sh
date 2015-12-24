for i in `seq 1 1`
do
	#hadoop fs -rm -r spark_input
	#hadoop fs -rm -r spark_output
	#spark-submit --class com.spark.example.SparkPi --master yarn-client --executor-memory 1G --executor-gpu-memory 256MB target/SparkPi-1.4.1.jar 2
	spark-submit --class com.spark.example.SparkPi --master yarn-client --executor-memory 1G target/SparkPi-1.4.1.jar 8
done
