package com.example.hadoop;

import com.example.input.WholeFileInputFormat;
import com.example.jni.CudaWrapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.StringTokenizer;


public class StringMatch extends Configured implements Tool {
	private static final String MATCHINGWORD = "ipsum";

	public static class MyMapper extends Mapper<Object, BytesWritable , Text, IntWritable> {
		private static final IntWritable ONE = new IntWritable(1);
		private final transient Text word = new Text(MATCHINGWORD);

		private Path[] localFiles;
		private boolean defaultRunFlag = true;

		public void map(Object key, BytesWritable value, Context context)
			throws IOException, InterruptedException {
			int i = 0;
			byte[] bytes = value.getBytes();

			// CPU Exec
			if(context.getConfiguration().getBoolean("GPU_FLAG", defaultRunFlag) == false){
				System.out.println("***MapTask(CPU): " + context.getTaskAttemptID().getTaskID() + " ***");

				String str = new String(bytes);
				StringTokenizer itr = new StringTokenizer(str.toLowerCase(), " ,.\n\0");
				String tmp;

				//System.out.println("Input: ");
				while (itr.hasMoreTokens()) {
					tmp = itr.nextToken();
					//System.out.println(tmp);
					if(tmp.equals(MATCHINGWORD)) {
						context.write(word, ONE);
					}
				}
				//System.out.println("End: ");
			}

			// GPU Exec
			else if(context.getConfiguration().getBoolean("GPU_FLAG", defaultRunFlag) == true) {
				System.out.println("***MapTask(GPU): " + context.getTaskAttemptID().getTaskID() + " ***");
				System.out.println("***GPU Device >> " + context.getGpuDeviceID());
				//System.out.println("***GPU Device >> " + context.getTaskAttemptID().getTaskID().getId()%2);

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
				System.setProperty("java.library.path", libPath);

				//Call CUDA
				CudaWrapper m = new CudaWrapper(libPath);

				// call the native method, which in turn will execute kernel code on the device
				int count = m.CUDAProxy_stringMatch(bytes, MATCHINGWORD, context.getGpuDeviceID());
				System.out.println("***Count >> " + count);
				context.write(word, new IntWritable(count));
			}
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
			sum += val.get();
		}
			System.out.println(MATCHINGWORD+ " >> " + sum);
			context.write(key, new IntWritable(sum));
		}
	}

	@Override
	public int run(final String[] args) throws Exception {
		final Configuration conf = this.getConf();
		final Job job = Job.getInstance(conf, "Matrix Calculation");
		boolean gpuFlag = false;

		job.setJarByClass(StringMatch.class);

		job.setMapperClass(MyMapper.class);
		job.setReducerClass(MyReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		//job.setInputFormatClass(TextInputFormat.class);
		job.setInputFormatClass(WholeFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		//job.setNumReduceTasks(0);

		if (args.length < 3) {
			System.err.println("Usage: stringMatch <in> <out> <CPU/GPU>");
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

		DistributedCache.addCacheFile(new Path("/user/gpu/native/program.so").toUri(), job.getConfiguration());
		//DistributedCache.addCacheFile(new Path("/home/gpu/workspace/hadoop_app/WordCount/src/jni/program.so").toUri(), job.getConfiguration());

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(final String[] args) throws Exception {
		final int returnCode = ToolRunner.run(new Configuration(), new StringMatch(), args);
		System.exit(returnCode);
	}
}
