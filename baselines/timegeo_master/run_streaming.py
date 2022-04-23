import os
import sys

hadoop_jar = "hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.5.jar"
queue_name = "-D mapreduce.job.queuename='usera' -D mapreduce.job.name='fengjie:streaming'"
file_path = "code"

mapred_reduce = "-D mapreduce.job.reduces=0"
mapper = "-mapper 'python code/pre_map.py'"
reducer = ""

