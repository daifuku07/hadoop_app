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

		public static int SIZE = 5;

		@Override public void map(final LongWritable key, final Text value, final Context context)
			throws IOException, InterruptedException {
			final String line = value.toString();

			//Load Shared Library
			localFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());

			String libPath = "";
			if (null != localFiles)
			{
				System.out.println("***localFiles.length: " + localFiles.length);
				if (localFiles.length > 0)
				{
					for (int i = 0; i < localFiles.length; i++)
					{
						Path localFile = localFiles[i];
						System.out.println("***local file: " + localFile);

						if(localFile.toString().endsWith("program.so"))
						{
							libPath = localFile.toString();
						}
					}
				}
			}
			else
			{
				System.out.println("***localFiles was null!");
				return;
			}


			System.setProperty("java.library.path", libPath);

			//Call CUDA
			CudaWrapper m = new CudaWrapper(libPath);

			float[] a = new float[SIZE * SIZE];
			float[] b = new float[SIZE * SIZE];
			float[] c = new float[SIZE * SIZE];
			// initialize two arrays
			for (int i = 0; i < SIZE; i++){
				for(int j = 0; j < SIZE; j++){
					a[i * SIZE + j] = i; 
					b[i * SIZE + j] = i;
				}
			}
			System.out.println("J: Arrays initialized, calling C. Size = " + a.length);
			
			// call the native method, which in turn will execute kernel code on the device
			System.out.println("J:calling C.");
			int retVal = m.CUDAProxy_matrixMul(a, b, c, SIZE);
			System.out.println("J: retVal = \nJ:c[]= " + retVal);
			
			// print the results
			for (int i = 0; i < retVal; i++)
				System.out.print("J: " + c[i] + "| ");
			System.out.println();

			word.set(Arrays.toString(c));
			context.write(word, ONE);

//			while (tokenizer.hasMoreTokens()) {
//				word.set(tokenizer.nextToken());
//				context.write(word, ONE);
//			}
		}
	}


	public static class MyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

		@Override
		public void reduce(final Text key, final Iterable<IntWritable> values, final Context context)
		throws IOException, InterruptedException {
		int sum = 0;
		for (final IntWritable val : values) {
			sum += val.get();
		}
		context.write(key, new IntWritable(sum));
		}
	}

	@Override public int run(final String[] args) throws Exception {
		final Configuration conf = this.getConf();
		final Job job = Job.getInstance(conf, "Matrix Calculation");
		job.setJarByClass(MatrixCalc.class);

		job.setMapperClass(MyMapper.class);
		job.setReducerClass(MyReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		//Load CUDA shared Library
		FileSystem fs = FileSystem.get(conf);

		DistributedCache.addCacheFile(new Path("/user/gpu/native/program.so").toUri(), job.getConfiguration());
		//DistributedCache.addCacheFile(new Path("/home/gpu/workspace/hadoop_app/WordCount/src/jni/program.so").toUri(), job.getConfiguration());

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(final String[] args) throws Exception {
		final int returnCode = ToolRunner.run(new Configuration(), new MatrixCalc(), args);
		System.exit(returnCode);
	}
}
