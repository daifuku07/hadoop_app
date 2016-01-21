package com.example.hadoop;

import com.example.jni.CudaWrapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.Arrays;
import java.util.StringTokenizer;


public class MatrixCalc extends Configured implements Tool {

	public static class MyMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
		private static final IntWritable ONE = new IntWritable(1);
		private final transient Text word = new Text();
		private Path[] localFiles;

		@Override
		public void map(final LongWritable key, final Text value, final Context context)
			throws IOException, InterruptedException {
			//final String line = value.toString();
			//final StringTokenizer tokenizer = new StringTokenizer(line);

			int i, j, k;
			int num = 0;

			//500MB
			//int size = 400 * 1024 *1024;
			int size = 450 * 1024 *1024;

			int[] a = new int[size/4];

			for(i = 0; i < (size/4); i++) {
				a[i] = i;
			}

			System.out.println("***MapTask(GPU): " + context.getTaskAttemptID().getTaskID() + " ***");
			System.out.println("***GPU Device >> " + context.getTaskAttemptID().getTaskID().getId()%2);
			System.out.println("***size(" + (size/1024/1024) + "): ");

			//Load Shared Library
			localFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());

			String libPath = "";
			if (null != localFiles) {
				if (localFiles.length > 0) {
					for (i = 0; i < localFiles.length; i++) {
						Path localFile = localFiles[i];
						if (localFile.toString().endsWith("program.so")) {
							libPath = localFile.toString();
						}
					}
				}
			} else {
				System.out.println("***localFiles was null!");
				return;
			}

			System.out.println("J:java.library.path = " + libPath);
			System.setProperty("java.library.path", libPath);

			//Call CUDA
			CudaWrapper m = new CudaWrapper(libPath);

			// call the native method, which in turn will execute kernel code on the device
			System.out.println("J:calling C.");
			int retVal = m.CUDAProxy_matrixMul(a, size, 1);
			System.out.println("J: retVal = \nJ:c[]= " + retVal);

			word.set(Integer.toString(retVal));
			context.write(word, ONE);
		}
	}

	public static class MyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(final Text key, final Iterable<IntWritable> values, final Context context)
		throws IOException, InterruptedException {
		int sum = 0;
		System.out.println("***reduce");
//		System.out.println("***GPU Device >> " + context.getGpuDeviceID());

		for (final IntWritable val : values) {
			System.out.println("***reduce calc");
			sum += val.get();
		}
		context.write(key, new IntWritable(sum));
		}
	}

	@Override public int run(final String[] args) throws Exception {
		final Configuration conf = this.getConf();
		final Job job = Job.getInstance(conf, "Matrix Calculation");
		boolean gpuFlag = false;

		job.setJarByClass(MatrixCalc.class);

		job.setMapperClass(MyMapper.class);
		job.setReducerClass(MyReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setNumReduceTasks(0);

		if (args.length < 3) {
			System.err.println("Usage: matrixMul <in> <out> <CPU/GPU>");
			System.exit(2);
		}

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		if(args[2].equals("CPU")){
			gpuFlag = false;
			System.out.println("***Choose CPU Mode");
		}
		else if(args[2].equals("GPU")){
			gpuFlag = true;
			System.out.println("***Choose GPU Mode");
		}
		else{
			System.err.println("ERROR: <CPU/GPU>");
			System.exit(2);
		}
		// Set GPU FLAG
		conf.setBoolean("GPU_FLAG", gpuFlag);

		//Load CUDA shared Library
		FileSystem fs = FileSystem.get(conf);

		System.out.println("J:java.library.path = " + new Path("/user/master/native/program.so").toUri());
		DistributedCache.addCacheFile(new Path("/user/master/native/program.so").toUri(), job.getConfiguration());
		//DistributedCache.addCacheFile(new Path("/home/master/workspace/hadoop_app/WordCount/src/jni/program.so").toUri(), job.getConfiguration());

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(final String[] args) throws Exception {
		final int returnCode = ToolRunner.run(new Configuration(), new MatrixCalc(), args);
		System.exit(returnCode);
	}
}
