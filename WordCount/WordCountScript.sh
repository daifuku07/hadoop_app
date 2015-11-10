hadoop fs -put src/jni native
hadoop fs -put inFiles input
hadoop jar out/artifacts/WordCount_jar/WordCount.jar hadoop.WordCount input output
hadoop fs -rm -r output
hadoop jar out/artifacts/WordCount_jar/WordCount.jar hadoop.WordCount input output
hadoop fs -rm -r output
hadoop jar out/artifacts/WordCount_jar/WordCount.jar hadoop.WordCount input output
hadoop fs -rm -r output
hadoop jar out/artifacts/WordCount_jar/WordCount.jar hadoop.WordCount input output
