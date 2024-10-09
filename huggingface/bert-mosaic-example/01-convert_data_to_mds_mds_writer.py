# Databricks notebook source
# MAGIC %pip install -U mosaicml-streaming mosaicml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Mosaic Streaming 
# MAGIC - Example doc: https://docs.mosaicml.com/projects/streaming/en/stable/how_to_guides/synthetic_nlp.html
# MAGIC - End-to-end example: https://github.com/mosaicml/examples/blob/main/examples/end-to-end-examples/sec_10k_qa/convert_10ks_to_mds.py

# COMMAND ----------

# MAGIC %md ### 1. Convert Spark dataframe to MDS

# COMMAND ----------

import transformers
import datasets
import os
from multiprocessing import cpu_count
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType

tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

# Create BERT tokenizer
def tokenize_function(sample):
    return tokenizer(
        text=sample['sentence'],
        padding="max_length",
        max_length=256,
        truncation=True
    )

# Tokenize SST-2
sst2_dataset = datasets.load_dataset("glue", "sst2", num_proc=os.cpu_count() - 1)
tokenized_sst2_dataset = sst2_dataset.map(tokenize_function,
                                          batched=True,
                                          num_proc=cpu_count(),
                                          batch_size=100,
                                          remove_columns=['idx', 'sentence'])

# Split dataset into train and validation sets
train_dataset = tokenized_sst2_dataset["train"]
eval_dataset = tokenized_sst2_dataset["validation"]

# Define schema
schema = StructType([
    StructField("label", IntegerType(), True),
    StructField("input_ids", ArrayType(IntegerType()), True),
    StructField("token_type_ids", ArrayType(IntegerType()), True),
    StructField("attention_mask", ArrayType(IntegerType()), True)
])

# Convert to Spark DataFrame
spark_train_dataset = spark.createDataFrame(train_dataset.to_pandas(), schema=schema)
spark_eval_dataset = spark.createDataFrame(eval_dataset.to_pandas(), schema=schema)

spark_train_dataset.display()

# COMMAND ----------

# spark_train_dataset.write.saveAsTable("main.alex_m.glue_sst2_dataset")
# spark_train_dataset.write.format("parquet").save("/Volumes/main/alex_m/my_volume/parquet/spark_train")

# COMMAND ----------

from streaming.base.converters import dataframe_to_mds
from streaming.base import MDSWriter
from shutil import rmtree
import os

# Parameters required for saving data in MDS format
columns = {
    'label': 'int32',
    'input_ids': 'ndarray:int32',
    'token_type_ids': 'ndarray:int32',
    'attention_mask': 'ndarray:int32'
    }

# compression algorithms
compression = 'zstd:7'
hashes = ['sha1']
limit = 8192

# Specify where the data will be stored
out_root = '/Volumes/main/alex_m/my_volume/mds-text/glue-sst2'
output_dir_train = os.path.join(out_root, 'spark_train')
output_dir_validation = os.path.join(out_root, 'spark_validation')

# Save the training data using the `dataframe_to_mds` function, which divides the dataframe into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=4):
    if os.path.exists(output_path):
        print(f"Deleting {label} data: {output_path}")
        rmtree(output_path)
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {'out': output_path, 'columns': columns, 'compression': compression, 'hashes': hashes, 'size_limit': limit}
    dataframe_to_mds(df.repartition(num_workers), merge_index=True, mds_kwargs=mds_kwargs)

# save full dataset
save_data(spark_train_dataset, output_dir_train, 'train', 10)
save_data(spark_eval_dataset, output_dir_validation, 'validation', 10)

# COMMAND ----------

# MAGIC %md ### 2. Example that saves the text/sentences (not tokenizer)

# COMMAND ----------

import os
import json
from glob import glob
from typing import Iterator, Tuple
from multiprocessing import Pool
from streaming import MDSWriter, StreamingDataset
from datasets import load_dataset

# COMMAND ----------

out_root = '/Volumes/main/alex_m/my_volume/mds-text/glue-sst2'
num_groups = 1
num_process = os.cpu_count()

# COMMAND ----------

def each_task(out_root: str, groups: int, total_samples: int) -> Iterator[Tuple[str, int, int]]:
    samples_per_group = total_samples // groups
    for data_group in range(groups):
        sub_out_root = os.path.join(out_root, str(data_group))
        start_sample_idx = data_group * samples_per_group
        end_sample_idx = start_sample_idx + samples_per_group - 1 if data_group < groups - 1 else total_samples - 1
        yield sub_out_root, start_sample_idx, end_sample_idx

def convert_to_mds(args: Tuple[str, int, int, dict]) -> None:
    sub_out_root, start_sample_idx, end_sample_idx, dataset = args
    
    columns = {
        'sentence': 'str',
        'label': 'int',
        'idx': 'int'
    }
    
    with MDSWriter(out=sub_out_root, columns=columns) as out:
        for i in range(start_sample_idx, end_sample_idx + 1):
            sample = dataset[i]
            out.write({
                'sentence': sample['sentence'],
                'label': sample['label'],
                'idx': sample['idx']
            })

def init_worker():
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')

# COMMAND ----------

# Clean up root directory
os.system(f'rm -rf {out_root}')

# Load the dataset once
dataset = datasets.load_dataset("glue", "sst2", num_proc=os.cpu_count() - 1)
total_samples = len(dataset)

# Prepare arguments for each task
arg_tuples = [(sub_out_root, start_idx, end_idx, dataset) 
              for sub_out_root, start_idx, end_idx in each_task(out_root, groups=num_groups, total_samples=total_samples)]

# Process group of data in parallel into directories of shards
with Pool(initializer=init_worker, processes=num_process) as pool:
    pool.map(convert_to_mds, arg_tuples)

print('Finished conversion')

# Merge metadata
from streaming.base.util import merge_index
merge_index(out_root, keep_local=True)
print('Merged index files')

# COMMAND ----------

# Load and verify MDS dataset
mds_dataset = StreamingDataset(local=out_root, remote=None, shuffle=False, batch_size=10)

for ix, sample in enumerate(mds_dataset):
    if ix < 5:  # Print first 5 samples
        print(f"Sample {ix}:")
        print(f"Sentence: {sample['sentence']}")
        print(f"Label: {sample['label']}")
        print(f"Index: {sample['idx']}")
        print()
    else:
        break

print(f"Total samples in MDS dataset: {len(mds_dataset)}")

# COMMAND ----------

# MAGIC %md ### 3. Group raw data, tokenize and convert to MDS format in parallel
# MAGIC - https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/parallel_dataset_conversion.html#2.-Group-the-raw-data-and-convert-to-MDS-format-in-parallel

# COMMAND ----------

import os
from typing import Iterator, Tuple
from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from streaming import MDSWriter, StreamingDataset
from transformers import AutoTokenizer
import numpy as np

# Global settings
out_root = '/Volumes/main/alex_m/my_volume/mds-text/glue-sst2'
num_groups = 1
num_process = os.cpu_count()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(sample):
    return tokenizer(
        text=sample['sentence'],
        padding="max_length",
        max_length=256,
        truncation=True
    )

def load_and_tokenize_hf_dataset():
    dataset = load_dataset("glue", "sst2")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cpu_count(),
        batch_size=100,
        remove_columns=['idx', 'sentence']
    )
    return tokenized_dataset['train'], tokenized_dataset['validation']

def each_task(out_root: str, groups: int, total_samples: int) -> Iterator[Tuple[str, int, int]]:
    samples_per_group = total_samples // groups
    for data_group in range(groups):
        sub_out_root = os.path.join(out_root, str(data_group))
        start_sample_idx = data_group * samples_per_group
        end_sample_idx = start_sample_idx + samples_per_group - 1 if data_group < groups - 1 else total_samples - 1
        yield sub_out_root, start_sample_idx, end_sample_idx

def convert_to_mds(args: Tuple[str, int, int, dict]) -> None:
    sub_out_root, start_sample_idx, end_sample_idx, dataset = args
    
    columns = {
        'input_ids': 'ndarray',
        'token_type_ids': 'ndarray',
        'attention_mask': 'ndarray',
        'label': 'int'
    }
    
    with MDSWriter(out=sub_out_root, columns=columns) as out:
        for i in range(start_sample_idx, end_sample_idx + 1):
            sample = dataset[i]
            out.write({
                'input_ids': np.array(sample['input_ids'], dtype=np.int32),
                'token_type_ids': np.array(sample['token_type_ids'], dtype=np.int32),
                'attention_mask': np.array(sample['attention_mask'], dtype=np.int32),
                'label': sample['label']
            })

def init_worker():
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')

def process_split(split_name, dataset, out_root):
    split_out_root = os.path.join(out_root, split_name)
    os.makedirs(split_out_root, exist_ok=True)
    
    total_samples = len(dataset)
    arg_tuples = [(sub_out_root, start_idx, end_idx, dataset) 
                  for sub_out_root, start_idx, end_idx in each_task(split_out_root, groups=num_groups, total_samples=total_samples)]

    with Pool(initializer=init_worker, processes=num_process) as pool:
        pool.map(convert_to_mds, arg_tuples)

    from streaming.base.util import merge_index
    merge_index(split_out_root, keep_local=True)
    print(f'Finished conversion for {split_name} split')

# Clean up root directory
os.system(f'rm -rf {out_root}')
os.makedirs(out_root, exist_ok=True)

# Load and tokenize the dataset
train_dataset, val_dataset = load_and_tokenize_hf_dataset()

# Process train split
process_split('train', train_dataset, out_root)

# Process validation split
process_split('validation', val_dataset, out_root)

print('Finished all conversions')

# Load and verify MDS datasets
for split in ['train', 'validation']:
    split_out_root = os.path.join(out_root, split)
    mds_dataset = StreamingDataset(local=split_out_root, remote=None, shuffle=False, batch_size=10)

    print(f"\nVerifying {split} split:")
    for ix, sample in enumerate(mds_dataset):
        if ix < 5:  # Print first 5 samples
            print(f"Sample {ix}:")
            print(f"Input IDs: {sample['input_ids'].shape}, {sample['input_ids'].dtype}")
            print(f"First few elements: {sample['input_ids'][:5]}")
            print(f"Token Type IDs: {sample['token_type_ids'].shape}, {sample['token_type_ids'].dtype}")
            print(f"First few elements: {sample['token_type_ids'][:5]}")
            print(f"Attention Mask: {sample['attention_mask'].shape}, {sample['attention_mask'].dtype}")
            print(f"First few elements: {sample['attention_mask'][:5]}")
            print(f"Label: {sample['label']}")
            print()
        else:
            break

    print(f"Total samples in {split} MDS dataset: {len(mds_dataset)}")
