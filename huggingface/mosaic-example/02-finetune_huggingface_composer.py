# Databricks notebook source
# MAGIC %md ## Finetuning Hugging Face Models using Composer, Mosaic Streaming, and TorchDistributor
# MAGIC - References Mosaic Composer documentation: https://docs.mosaicml.com/projects/composer/en/stable/examples/finetune_huggingface.html
# MAGIC - Composer Trainer API: https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html

# COMMAND ----------

# MAGIC %md This notebook converts the Mosaic Composer HuggingFace example on their documentation over to Databricks. 
# MAGIC
# MAGIC This demo will show you how to train Hugging Face model using Composer, Mosaic Streaming, and TorchDistributor (Hugging Face model and Trainer) for single-node + single-GPU, single-node + multi-GPU, and multi-node + multi-GPU. 
# MAGIC
# MAGIC The cluster configurations for this demo are as follows but it's important to note that TorchDistributor does not use the driver for training model when using the multi-node + multi-GPU scenario. 
# MAGIC ```
# MAGIC {
# MAGIC     "spark_version": "15.4.x-gpu-ml-scala2.12",
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 1,
# MAGIC         "availability": "SPOT_WITH_FALLBACK",
# MAGIC         "zone_id": "auto",
# MAGIC         "spot_bid_price_percent": 100
# MAGIC     },
# MAGIC     "node_type_id": "g5.12xlarge",
# MAGIC     "autotermination_minutes": 120,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "effective_spark_version": "15.4.x-gpu-ml-scala2.12",
# MAGIC     "num_workers": 2,
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC You can create a cluster that has different driver instance type vs. worker nodes through API (instructions below)
# MAGIC ```
# MAGIC clusters create --json '{
# MAGIC   "cluster_name": "xxxx",
# MAGIC   "spark_version": "14.3.x-gpu-ml-scala2.12",
# MAGIC   "node_type_id": "g4dn.12xlarge",
# MAGIC   "driver_node_type_id": "i3.xlarge",
# MAGIC   "autoscale" : { "min_workers": 1, "max_workers": 2 },
# MAGIC   "aws_attributes" : {"first_on_demand": 3} 
# MAGIC }'
# MAGIC ```
# MAGIC
# MAGIC Make sure you instantiate/download the Hugging Face model outside of the `main` training function or else TorchDistributor will fail due to memory leakage.
# MAGIC
# MAGIC Using Composer's [MLflow logger](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.MLFlowLogger.html#composer.loggers.MLFlowLogger) to track model and system metrics but still working with Eng on best way to log checkpoints. Right now, I have a work around to copy checkpoints from local_disk0 to the correct MLflow run

# COMMAND ----------

# MAGIC %pip install -U mosaicml-streaming mosaicml peft
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Arguments used in training function

# COMMAND ----------

import os
from dataclasses import dataclass, field
from enum import Enum

out_root = '/Volumes/main/alex_m/my_volume/mds-text/glue-sst2'
train_root = os.path.join(out_root, 'train')
eval_root = os.path.join(out_root, 'validation')
local_dir = '/local_disk0/composer-training'

@dataclass
class Args:
    """
    Training arguments.
    """
    batch_size: int = 64 # Set a larger batch size due to the large size of dataset
    train_subset_num_batches: int = -1
    max_duration: str = "1ep"
    save_folder: str = "/local_disk0/checkpoints"
    save_interval: str = "1ep"
    save_overwrite: bool = False
    precision: str = "amp_fp16"  # 'amp' for mixed precision training, can provide 2x speed on NVIDIA GPUs

class TrainingMethod(str, Enum):
    SNSG = "Single Node Single GPU Training"
    SNMG = "Single Node Multi GPU Training"
    MNMG = "Multi Node Multi GPU Training"

# TODO: Specify what level of distribution will be used for training. The Single-Node Multi-GPU and Multi-Node Multi-GPU arrangements will use the TorchDistributor for training.
training_method = TrainingMethod.MNMG

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

import uuid
import os

from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy
from streaming import StreamingDataset, StreamingDataLoader
import streaming.base.util as util
import transformers

def get_model(model_name='bert-base-uncased', num_classes=2):
    # Create a BERT sequence classification model using Hugging Face transformers
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=num_classes, average='micro')]
    # Package as a trainer-friendly Composer model
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    
    return composer_model

def get_dataloader_with_mosaic(path, batch_size, label, use_local=False):

    random_uuid = uuid.uuid4()
    local_path = f"/local_disk0/{random_uuid}"
    print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
    if use_local:
        dataset = StreamingDataset(remote=path, local=local_path, shuffle=False, batch_size=batch_size)
    else:
        dataset = StreamingDataset(local=path, shuffle=False, batch_size=batch_size)
    data_collator = transformers.data.data_collator.default_data_collator
    return StreamingDataLoader(dataset, 
                               batch_size=batch_size, 
                               shuffle=False, 
                               collate_fn=data_collator,
                               pin_memory=True,
                               num_workers=4,
                               drop_last=True)

import os
import mlflow

def log_checkpoints_to_mlflow(run_id, local_checkpoint_dir):
    import os
    import mlflow
    # Set up MLflow client
    client = mlflow.tracking.MlflowClient()

    # Get all .pt files
    pt_files = [f for f in os.listdir(local_checkpoint_dir) if f.endswith('.pt')]

    # Log file sizes (optional)
    for f in pt_files:
        file_path = os.path.join(local_checkpoint_dir, f)
        file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
        print(f"File: {f}, Size: {file_size_gb:.2f} GB")

    # Check if there's an active run with the given run_id
    try:
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            print(f"Using existing active run with ID: {run_id}")
            run_context = mlflow.active_run()
        else:
            print(f"Starting new run with ID: {run_id}")
            run_context = mlflow.start_run(run_id=run_id)
    except Exception as e:
        print(f"Error checking active run: {e}")
        print(f"Starting new run with ID: {run_id}")
        run_context = mlflow.start_run(run_id=run_id)

    # Log artifacts to MLflow
    with run_context:
        for f in pt_files:
            local_path = os.path.join(local_checkpoint_dir, f)
            artifact_path = "checkpoints"
            client.log_artifact(run_id, local_path, artifact_path)
            print(f"Logged {f} to MLflow artifacts")

    print("Finished logging checkpoints to MLflow")


def save_and_log_hf_checkpoints(checkpoint_save_folder, run_id, local_checkpoint_dir):
    """
    Saves Composer checkpoints as HuggingFace checkpoints and logs them to MLflow.

    Args:
    args: An object containing the save_folder attribute.
    run_id: The MLflow run ID to log artifacts to.
    """
    import os
    from pathlib import Path
    import tempfile
    import mlflow
    import composer

    # with tempfile.TemporaryDirectory() as temp_dir:
    #     temp_output = Path(temp_dir) / "temp_output"
    #     temp_output.mkdir(exist_ok=True)

    # Write composer checkpoints to hf checkpoints
    print(f"Saving Composer checkpoints from {checkpoint_save_folder} to HuggingFace checkpoints {local_checkpoint_dir}")
    composer.models.write_huggingface_pretrained_from_composer_checkpoint(
        Path(checkpoint_save_folder), 
        output_folder=Path(local_checkpoint_dir),
    )
    # composer.models.write_huggingface_pretrained_from_composer_checkpoint(
    #     Path(checkpoint_save_folder) / "latest-rank0.pt", 
    #     output_folder=Path(local_checkpoint_dir),
    # )

    # Set up MLflow client
    client = mlflow.tracking.MlflowClient()

    hf_files = os.listdir(local_checkpoint_dir)
    # with mlflow.start_run(run_id=run_id):
    for hf_file in hf_files:
        print(f"Logging {hf_file} to MLflow artifacts")
        client.log_artifact(
            run_id, 
            str(Path(local_checkpoint_dir) / hf_file),
            "hf_checkpoints"
        )

def get_run_id_by_name(experiment_id, run_name):

    import mlflow
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
    if runs:
        return runs[0].info.run_id
    else:
        return None

  
# test data loader
train_dataloader = get_dataloader_with_mosaic(train_root, batch_size=16, label="train")
for ix, sample in enumerate(train_dataloader):
  if ix < 5:  # Print first 5 samples
      print(f"Sample {ix}:")
      print(f"Sample: {sample}")
      print(f"Input IDs: {sample['input_ids'].shape}, {sample['input_ids'].dtype}")
      print(f"First few elements: {sample['input_ids'][:5]}")
      print(f"Token Type IDs: {sample['token_type_ids'].shape}, {sample['token_type_ids'].dtype}")
      print(f"First few elements: {sample['token_type_ids'][:5]}")
      print(f"Attention Mask: {sample['attention_mask'].shape}, {sample['attention_mask'].dtype}")
      print(f"First few elements: {sample['attention_mask'][:5]}")
      print(f"Label: {sample['labels']}")
  else:
      break

print(f"Total samples in MDS dataset: {len(train_dataloader)}")

# COMMAND ----------

# MAGIC %md ### Setup MLflow

# COMMAND ----------

import mlflow
import os

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/composer-streaming-bert'
 
# You will need these later
db_host = os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md ### Set Arguments for each training method

# COMMAND ----------

args = Args(
    max_duration="1ep", 
    batch_size=64,
    train_subset_num_batches=10,
    save_overwrite=True,
    save_folder='/local_disk0/composer-training/checkpoints'
  )

# COMMAND ----------

# MAGIC %md ### Train BERT model on single-node GPU using Mosaic Streaming

# COMMAND ----------

import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from streaming import StreamingDataset, StreamingDataLoader
import streaming.base.util as util

from torchmetrics.classification import MulticlassAccuracy

from composer import Trainer
import composer
from composer.callbacks import CheckpointSaver
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy

# update training method to train single-node, single-GPU
training_method = TrainingMethod.SNSG
if training_method == TrainingMethod.SNSG:
    # get dataloader
    train_dataloader = get_dataloader_with_mosaic(train_root, batch_size=args.batch_size, label="train")
    eval_dataloader = get_dataloader_with_mosaic(eval_root, batch_size=args.batch_size, label="eval")

    # get model
    composer_model = get_model()

    # create optimizer
    optimizer = AdamW(
        params=composer_model.parameters(),
        lr=3e-5, betas=(0.9, 0.98),
        eps=1e-6, weight_decay=3e-6
    )
    linear_lr_decay = LinearLR(
        optimizer, start_factor=1.0,
        end_factor=0, total_iters=150
    )
    # setup mlflow logger
    mlflow_logger = composer.loggers.MLFlowLogger(experiment_name=experiment_path,
                                                    synchronous=True,
                                                    tracking_uri="databricks",
                                                    resume=True)
    

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        train_subset_num_batches=args.train_subset_num_batches,
        max_duration=args.max_duration,
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        save_folder=args.save_folder,
        save_overwrite=args.save_overwrite,
        device="gpu",
        precision=args.precision, # 'amp' for mixed precision training, can provide 2x spped on NVIDIA GPUs
        seed=17,
        loggers=[mlflow_logger]
    )

    # Start training
    trainer.fit()

    # log pt & hf checkpoints to mlflow
    run_id = get_run_id_by_name(experiment.experiment_id, mlflow_logger.run_name)
    log_checkpoints_to_mlflow(run_id, args.save_folder)
    save_and_log_hf_checkpoints(args.save_folder, run_id, "/local_disk0/hf_checkpoints")
    
    mlflow.end_run()

else:
    print(f"Training method was set to {training_method}")

# COMMAND ----------

# DBTITLE 1,Run inference using PyTorch state dict
# get model architecture
model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# run_id and latest-rank0.pt that we saved in mlflow checkpoints
# run_id = "3f4fef2a20bb4cb5955793e08573795e"
artifact_path = "checkpoints/latest-rank0.pt"

# Load the state dict from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
state_dict = torch.load(local_path, map_location=device)
model.load_state_dict(state_dict['state']['model'])

# Set the model to evaluation mode
model.eval()

# run inference on eval_batch
eval_batch = next(iter(eval_dataloader))

# Move batch to the same device as the model
eval_batch = {k: v.to(device) for k, v in eval_batch.items()}

with torch.no_grad():
    predictions = model(eval_batch)["logits"].argmax(dim=1)

# Visualize only 5 samples
predictions = predictions[:5]

label = ['negative', 'positive']
for i, prediction in enumerate(predictions):
    # sentence = sst2_dataset["validation"][i]["sentence"]
    correct_label = label[eval_batch['labels'][i]]
    prediction_label = label[prediction]
    # print(f"Sample: {sentence}")
    print(f"Label: {correct_label}")
    print(f"Prediction: {prediction_label}")
    print()

# COMMAND ----------

# DBTITLE 1,Run Inference Using Hugging Face Checkpoints
artifact_path = "hf_checkpoints"

# Load the state dict from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(local_path, num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
model.eval()
model.to(device)

with torch.no_grad():
  output = model(**eval_batch)
  predictions = output["logits"].argmax(dim=1)

# Visualize only 5 samples
predictions = predictions[:5]

label = ['negative', 'positive']
for i, prediction in enumerate(predictions):
    # sentence = sst2_dataset["validation"][i]["sentence"]
    correct_label = label[eval_batch['labels'][i]]
    prediction_label = label[prediction]
    # print(f"Sample: {sentence}")
    print(f"Label: {correct_label}")
    print(f"Prediction: {prediction_label}")
    print()

# COMMAND ----------

from transformers import pipeline

hf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
hf_pipeline("that loves its characters and communicates something rather beautiful about human nature ")

# COMMAND ----------

text = "that loves its characters and communicates something rather beautiful about human nature "
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)
output = model(**encoded_input)
predictions = output["logits"].argmax(dim=1)
print(f"Raw output: {output}")
print(f"Prediction: {predictions}")

# COMMAND ----------

# MAGIC %md ### Train BERT model on single-node, multi-GPU

# COMMAND ----------

def main_fn(args):
    import uuid
    import os

    import mlflow
    import transformers
    import torch
    import torch.distributed as dist
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from torchmetrics.classification import MulticlassAccuracy
    from torch.utils.data import DataLoader

    from composer import Trainer
    import composer
    from composer.models.huggingface import HuggingFaceModel
    from composer.metrics import CrossEntropy

    from streaming import StreamingDataset, StreamingDataLoader
    import streaming.base.util as util
    from composer.utils import get_device

    # set environment variables for Databricks and TMPDIR for mlflow_logger
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    print("Running distributed training")
    dist.init_process_group("nccl")

    # mosaic streaming recommendations
    util.clean_stale_shared_memory()
    composer.utils.dist.initialize_dist(get_device(None))

    # load StreamingDataset and StreamingDataLoader
    train_dataloader = get_dataloader_with_mosaic(
        train_root, batch_size=args.batch_size, label="train"
    )
    eval_dataloader = get_dataloader_with_mosaic(
        eval_root, batch_size=args.batch_size, label="eval"
    )

    print("Creating mlflow logger..........\n")
    mlflow_logger = composer.loggers.MLFlowLogger(
        experiment_name=experiment_path,
        synchronous=True,
        resume=True,
        tracking_uri="databricks"
    )

    # Create Trainer Object
    print("Creating Composer Trainer\n")
    trainer = Trainer(
        model=composer_model,  # This is the model from the HuggingFaceModel wrapper class.
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        train_subset_num_batches=args.train_subset_num_batches,
        max_duration=args.max_duration,
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        save_folder=args.save_folder,
        save_interval=args.save_interval,
        save_overwrite=args.save_overwrite,
        device="gpu",
        precision=args.precision,  # 'amp' for mixed precision training, can provide 2x spped on NVIDIA GPUs
        loggers=[mlflow_logger],
    )
    # Start training
    print("Starting training\n")
    trainer.fit()


    import os
    from pathlib import Path
    import shutil
    import tempfile

    
    # this might be different between nodes but we dont really care the rank should be globally unique across all nodes
    rank_of_gpu = os.environ["RANK"]
    # local_file_path = Path(args.save_folder) / f"model.{rank_of_gpu}.pt"
    # local_file_path.parent.mkdir(parents=True, exist_ok=True)

    print("Getting MLflow run_id from mlflow logger")
    run_name = mlflow_logger.run_name
    run_id = get_run_id_by_name(experiment.experiment_id, run_name)

    # save model (state_dict) to local filesystem for only rank 0
    state_model_dict = trainer.state.model.state_dict()
    if rank_of_gpu == "0":
        # torch.save(state_model_dict, str(local_file_path))
        print(f"Logging checkpoints to mlflow run_id {run_id}")
        log_checkpoints_to_mlflow(run_id, args.save_folder)

    trainer.close()
   

    dist.destroy_process_group()

    return str(run_id)


# COMMAND ----------

# DBTITLE 1,Get model, optimizer and linear scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import MulticlassAccuracy

# get model
composer_model = get_model()

# create optimizer
optimizer = AdamW(
    params=composer_model.parameters(),
    lr=3e-5,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=3e-6,
)
linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0, end_factor=0, total_iters=150
)

# COMMAND ----------

# DBTITLE 1,Run single-node, multi-GPU
from pyspark.ml.torch.distributor import TorchDistributor
import torch.distributed as dist

args = Args(
    max_duration="1ep", 
    batch_size=64,
    train_subset_num_batches=10,  # use small number to test faster or -1 to use max_duration
    save_overwrite=True,
    save_folder='/local_disk0/composer-training/checkpoints2'
  )

# Set training method to run on single node, multi-GPU
training_method = TrainingMethod.SNMG
if training_method == TrainingMethod.SNMG:
    run_id = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(main_fn, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNMG)[1:-1]} to run training on this cell.")

from pathlib import Path

artifact_path = "checkpoints"

# Load the state dict from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
save_and_log_hf_checkpoints(Path(local_path) / "latest-rank0.pt", run_id, "/local_disk0/hf_checkpoints")

# COMMAND ----------

# MAGIC %md ### Train BERT model on multi-node, multi-GPU

# COMMAND ----------

args = Args(
    max_duration="1ep", 
    batch_size=64,
    train_subset_num_batches=-1, # use small number to test faster or -1 to use max_duration
    save_overwrite=True,
    save_folder='/local_disk0/composer-training/checkpoints3'
  )

# Set training method to run on single node, multi-GPU
training_method = TrainingMethod.MNMG
if training_method == TrainingMethod.MNMG:
    run_id = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True).run(main_fn, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNMG)[1:-1]} to run training on this cell.")

from pathlib import Path
artifact_path = "checkpoints"

# Load the state dict from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
save_and_log_hf_checkpoints(Path(local_path) / "latest-rank0.pt", run_id, "/local_disk0/hf_checkpoints")

# COMMAND ----------

# MAGIC %md ### Options for Batch Inference:
# MAGIC - Spark Pandas UDF example: https://docs.databricks.com/en/machine-learning/train-model/huggingface/model-inference-nlp.html
# MAGIC - Model Serving endpoint on GPUs with Spark Pandas UDF: https://docs.databricks.com/en/machine-learning/model-inference/batch-inference-throughput.html
# MAGIC - Ray LLM Batch Inference: https://docs.ray.io/en/latest/data/batch_inference.html
# MAGIC   - Setup Ray on Databricks: https://docs.databricks.com/en/machine-learning/ray/ray-create.html#fixed-size-ray-cluster
