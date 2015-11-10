package hadoop;//input format is whole file input - custom format

import input.WholeFileInputFormat;
import jni.CudaWrapper;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.net.URI;
import java.util.regex.Pattern;
import java.io.IOException;
import java.lang.InterruptedException;
import java.util.StringTokenizer;
import java.util.*;


public class WordCount extends Configured
  implements Tool {

    private static final Pattern UNDESIRABLES = Pattern.compile("[(){},.;!+\"?<>%]");

    public static class WCMapper
      extends Mapper<Object, BytesWritable , Text, IntWritable>{

        private Text filenameKey;
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private String elements[] = { "education", "politics", "sports", "agriculture" };
        private HashSet<String> dict = new HashSet<String>(Arrays.asList(elements));
        private IntWritable result = new IntWritable();

        private Path[] localFiles;
        private URI[] cacheFiles;

        public static int SIZE = 100;

        @Override
        protected void setup(Context context) throws IOException,
          InterruptedException {
            InputSplit split = context.getInputSplit();
            Path path = ((FileSplit) split).getPath();
            filenameKey = new Text(path.toString());
            //hello = HelloLib.INSTANCE;
        }

        public void map(Object key, BytesWritable value, Context context
        ) throws IOException, InterruptedException {
            byte[] bytes = value.getBytes();
            String str = new String(bytes);

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

                //IntWritable idx = new IntWritable(i);
                //context.write(new Text(localFile.toString()), idx);
              }
            }
          }
          else
          {
            System.out.println("***localFiles was null!");
            return;
          }


          System.setProperty("java.library.path", libPath);
          //System.load(libPath);

          System.out.println("Hello CUDA through JNI!");
          // make an instance of our class to access the native method

          CudaWrapper m = new CudaWrapper(libPath);
          //CudaWrapper m = new CudaWrapper("/home/gpu/workspace/hadoop_app/WordCount/src/jni/program.so");

//          for(int l = 0; l < 10; l ++) {
            // declare three arras
            float[] a = new float[SIZE];
            float[] b = new float[SIZE];
            float[] c = new float[SIZE];
            // initialize two arrays
            for (int i = 0; i < a.length; i++)
              a[i] = b[i] = i;
            System.out.println("J: Arrays initialized, calling C.");
            // call the native method, which in turn will execute kernel code on the device
            for (int p = 0; p < 100; p++) {
              System.out.println("J:calling C.");
              int retVal = m.CUDAProxy_matrixAdd(a, b, c);
              System.out.println("J: retVal = \nJ:c[]= " + retVal);
              // print the results
              for (int i = 0; i < retVal; i++)
                System.out.print("J: " + c[i] + "| ");
              System.out.println();
            }
//          }
            //default parsing - /n /t carriage return are the delimiters set
            StringTokenizer itr = new StringTokenizer(str.toLowerCase());
          while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
                //hello.hello();
            }
        }
    }

    public static class WCReducer
      extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }


    public int run(String[] args) throws Exception {

        // Configuration processed by ToolRunner
        Configuration conf = getConf();

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: wordcount <in> <out>");
            System.exit(2);
        }
        Job job = new Job(conf, "word count");
        if (job == null) {
            return -1;
        }

        job.setJarByClass(WordCount.class);
      job.setInputFormatClass(WholeFileInputFormat.class);
      job.setMapperClass(WCMapper.class);
      job.setNumReduceTasks(0);
      //job.setReducerClass(WCReducer.class);
        job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(IntWritable.class);
      FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
      FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

      FileSystem fs = FileSystem.get(conf);

      DistributedCache.addCacheFile(new Path("/user/gpu/native/program.so").toUri(), job.getConfiguration());
      //DistributedCache.addCacheFile(new Path("/home/gpu/workspace/hadoop_app/WordCount/src/jni/program.so").toUri(), job.getConfiguration());
      return job.waitForCompletion(true) ? 0 : 1;

    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new WordCount(), args);
        System.exit(exitCode);
    }
}