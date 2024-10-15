# Databricks notebook source
# DBTITLE 1,Create Volume if it doesn't exist
spark.sql("CREATE VOLUME IF NOT EXISTS main.alex_m.computer_vision")

# COMMAND ----------

# DBTITLE 1,Create directory to store fineweb
# MAGIC %sh
# MAGIC mkdir -p /Volumes/main/alex_m/computer_vision/coco

# COMMAND ----------

# DBTITLE 1,Extract list of files from shallow git clone
!cd /tmp && git clone --sparse https://huggingface.co/datasets/detection-datasets/coco coco_files && cd coco_files && git ls-tree -r --name-only origin/main | grep "data" > /Volumes/main/alex_m/computer_vision/coco/files.txt

# COMMAND ----------

# DBTITLE 1,DF to download files
from pyspark.sql.functions import concat, lit, col

url_df = spark.read.text("/Volumes/main/alex_m/computer_vision/coco/files.txt").toDF("path").\
    select(concat(lit("https://huggingface.co/datasets/detection-datasets/coco/resolve/main/"), col("path")).alias("url")) \
    # .repartition(1024) # repartition based on cluster size, higher repartitions the more files can be downloaded in parallel

display(url_df)

# COMMAND ----------

# MAGIC %md Here’s a utility function to download the files - most of it is extra code to record the exit status, and then loop and retry failures, but you could skip it if desired and just use the UDF inside:

# COMMAND ----------

# DBTITLE 1,Utility function to download files
import subprocess
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, lit
from delta.tables import DeltaTable

def download_all(url_df: DataFrame, download_path: str, temp_working_dir: str, cut_dirs: int = 0):
    """
    Downloads URLs given in a DataFrame in parallel. It's not a crawler, just a downloader.

    Args:
        url_df (DataFrame): A DataFrame of all complete URLs to download in one column called "url"
        download_path (string): The path to root dir to download files to. Should be a UC `/Volumes` or DBFS `/dbfs` path;
            that is, must be on distributed storage but must also be a local path
        temp_working_dir (string): A directory that may be used to store temporary Delta tables holding download status;
            must be on UC or DBFS and should be a path as readable to Spark (i.e. "dbfs:/tmp" or "/tmp" not "/dbfs/tmp")
        cut_dirs (int): When downloading, the path from the root of the URL path will be mirrored in the download path. This specifies
            how many levels of dirs should be ignored from that path. For example, if downloading http://foo.com/bar/bing/baz.html to 
            path /Volumes/buzz/, and cut_dirs=1, then the file will end up at /Volumes/buzz/bing/baz.html, omitting the "/bar" directory
    """

    url_df = url_df.select("url")

    # Get some healthy parallelism, maybe 10x cluster cores?
    target_min_parallelism = 10 * spark.sparkContext.defaultParallelism
    if url_df.rdd.getNumPartitions() < target_min_parallelism:
        print(f"Repartitioning up to {target_min_parallelism} partitions")
        url_df = url_df.repartition(target_min_parallelism)

    @udf('int')
    def do_download(url):
        print(url)
        wget_cmd = ["wget", "--no-host-directories", "--force-directories", f"--cut-dirs={cut_dirs}", 
                    "--timestamping", "--no-if-modified-since", "--no-show-progress", 
                    "--retry-on-http-error=503", "--waitretry=3", url]
        result = subprocess.run(wget_cmd, cwd=download_path, check=False, capture_output=True)
        if result.returncode != 0:
            print(result.stdout.decode('utf-8'))
            print(result.stderr.decode('utf-8'))
        return result.returncode
    
    try:
        work_dir_exists = len(dbutils.fs.ls(temp_working_dir)) > 0
    except:
        work_dir_exists = False

    if work_dir_exists:
        print(f"{temp_working_dir} exists, reusing state")
    else:
        print(f"{temp_working_dir} is empty, populating initial state")
        url_df.withColumn("status", lit(None).cast('int')).write.format('delta').save(temp_working_dir)

    no_success_status_filter = "status IS NULL OR status <> 0"

    print(f"Starting download to {download_path} using {temp_working_dir} for state")

    remaining = spark.read.format("delta").load(temp_working_dir).filter(no_success_status_filter).count()
    while remaining > 0:
        print(f"{remaining} URLs left to process")
        DeltaTable.forPath(spark, temp_working_dir).update(
            condition = no_success_status_filter,
            set = { "status": do_download("url") }
        )
        remaining = spark.read.format("delta").load(temp_working_dir).filter(no_success_status_filter).count()

    print(f"Complete. Deleting {temp_working_dir}")
    dbutils.fs.rm(temp_working_dir, recurse=True)

# COMMAND ----------

# DBTITLE 1,Run download_all
# download_all(url_df=url_df, download_path="/Volumes/main/alex_m/fineweb/data", temp_working_dir="/tmp/fineweb_download_state")
# "/Volumes/main/alex_m/computer_vision/coco/files.txt"
download_all(url_df=url_df, download_path="/Volumes/main/alex_m/computer_vision/coco", temp_working_dir="/tmp/coco_download_state")

# COMMAND ----------

final_df = spark.read.parquet("/Volumes/main/alex_m/computer_vision/coco/datasets/detection-datasets/coco/resolve/main/data/")
print(final_df.count())
display(final_df)
