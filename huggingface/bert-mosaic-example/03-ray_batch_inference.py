# Databricks notebook source
# MAGIC %md ## Using Ray Data for Batch Inference following [Ray PyTorch documentation example](https://docs.ray.io/en/latest/data/batch_inference.html)
# MAGIC
# MAGIC Spark cluster configurations. HAVE TO ADD `"spark.databricks.pyspark.dataFrameChunk.enabled": "true"` to Spark configurations to use Spark with Ray Data
# MAGIC ```
# MAGIC {
# MAGIC     "cluster_name": "Alex Miller's GPU Cluster",
# MAGIC     "spark_version": "15.4.x-gpu-ml-scala2.12",
# MAGIC     "spark_conf": {
# MAGIC         "spark.databricks.pyspark.dataFrameChunk.enabled": "true"
# MAGIC     },
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 1,
# MAGIC         "availability": "SPOT_WITH_FALLBACK",
# MAGIC         "zone_id": "auto",
# MAGIC         "spot_bid_price_percent": 100,
# MAGIC         "ebs_volume_count": 0
# MAGIC     },
# MAGIC     "node_type_id": "g5.12xlarge",
# MAGIC     "driver_node_type_id": "g5.12xlarge",
# MAGIC     "autotermination_minutes": 120,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "single_user_name": "alex.miller@databricks.com",
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "effective_spark_version": "15.4.x-gpu-ml-scala2.12",
# MAGIC     "num_workers": 2,
# MAGIC     "apply_policy_default_values": false
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### Write text data to delta table

# COMMAND ----------

# import datasets
# import os

# sst2_dataset = datasets.load_dataset("glue", "sst2", num_proc=os.cpu_count() - 1)

# # Convert each split to a Spark DataFrame
# train_df = spark.createDataFrame(sst2_dataset['train'].to_pandas())
# validation_df = spark.createDataFrame(sst2_dataset['validation'].to_pandas())
# test_df = spark.createDataFrame(sst2_dataset['test'].to_pandas())

# # Write each DataFrame to a Delta table
# train_df.write.format("delta").mode("overwrite").saveAsTable("main.alex_m.sst2_train_inference")
# validation_df.write.format("delta").mode("overwrite").saveAsTable("main.alex_m.sst2_validation_inference")
# test_df.write.format("delta").mode("overwrite").saveAsTable("main.alex_m.sst2_test_inference")

# COMMAND ----------

# MAGIC %md ### Setup Ray Cluster

# COMMAND ----------

# MAGIC %pip install -U mosaicml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, MAX_NUM_WORKER_NODES, shutdown_ray_cluster
import ray

restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass
  try:
    ray.shutdown()
  except:
    pass

# Ray allows you to define custom cluster configurations using setup_ray_cluster function
# This allows you to allocate CPUs and GPUs on Ray cluster
setup_ray_cluster(
  min_worker_nodes=2,       # minimum number of worker nodes to start
  max_worker_nodes=2,       # maximum number of worker nodes to start
  num_gpus_worker_node=4,   # number of GPUs to allocate per worker node
  num_gpus_head_node=4,     # number of GPUs to allocate on head node (driver)
  num_cpus_worker_node=40,  # number of CPUs to allocate on worker nodes
  num_cpus_head_node=8     # number of CPUs to allocate on head node (driver)
)


# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

# DBTITLE 1,Load Spark dataframe and convert to Ray dataset
import ray
import os


os.environ["DATABRICKS_HOST"] = "e2-demo-field-eng.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

ds = ray.data.read_databricks_tables(
  warehouse_id="475b94ddc7cd5211",
  table="sst2_train_inference",
  catalog="main",
  schema="alex_m"
)

# COMMAND ----------

ds.take(2)

# COMMAND ----------

ds.schema

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_env_vars

mlflow_db_creds = get_databricks_env_vars("databricks")

class HuggingFacePredictor:
    def __init__(self):
        import mlflow
        import torch
        import transformers
        from transformers import pipeline

        os.environ.update(mlflow_db_creds)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        artifact_path = "hf_checkpoints"
        run_id = "221a2295ccdd429c947eebde2bbeccc6"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(local_path, num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
        self.hf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=self.device)

    def __call__(self, batch):

        # batch["prediction"] = self.hf_pipeline(list(batch["sentence"]))
        sentences = list(batch["sentence"])
        batch["prediction"] = self.hf_pipeline(sentences, batch_size=len(sentences))
        return batch

# COMMAND ----------

predictions = ds.map_batches(
  HuggingFacePredictor,
  num_gpus=1,
  batch_size=100,
  concurrency=8
)

df = predictions.to_pandas()

# COMMAND ----------

df.shape
