# Databricks notebook source
# %pip install -q transformers accelerate timm
%pip install -q timm==1.0.9 albumentations==1.4.18 torchmetrics==1.4.3
%pip install -q -U pycocotools
dbutils.library.restartPython()

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
  num_cpus_head_node=40     # number of CPUs to allocate on head node (driver)
)


# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

import ray
import ray.train
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer

# COMMAND ----------

import json

ds = ray.data.read_parquet("/Volumes/main/alex_m/computer_vision/coco/datasets/detection-datasets/coco/resolve/main/data")

# Read the JSON file which contains metadata such as category names
with open('/Volumes/main/alex_m/computer_vision/coco/datasets/detection-datasets/coco/resolve/main/dataset_infos.json', 'r') as file:
    datataset_info = json.load(file)
categories = datataset_info['detection-datasets--coco']['features']['objects']['feature']['category']['names']
# convert the label and ids
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}
schema = ds.schema()
schema

# COMMAND ----------

ds.take(1)[0]

# COMMAND ----------

# MAGIC %md ### Convert bytes to PIL image

# COMMAND ----------

# ray.data.ActorPoolStrategy(size=3)

# COMMAND ----------

from PIL import Image
import io

def bytes_to_pil_image(batch):
    def convert_single_image(img_data):
        if isinstance(img_data, dict) and 'bytes' in img_data:
            return Image.open(io.BytesIO(img_data['bytes']))
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        else:
            raise ValueError(f"Unsupported image data type: {type(img_data)}")

    batch['image'] = batch['image'].apply(convert_single_image)
    return batch
  
processed_ds = ds.map_batches(
    bytes_to_pil_image,
    batch_format="pandas",
    batch_size=500,
    concurrency=3,
    num_gpus=4,
).limit(1000)

# processed_ds.take(1)

# COMMAND ----------

# MAGIC %md ### Show example of images (annotations, boxes, areas, labels)

# COMMAND ----------

import numpy as np
import os
from PIL import Image, ImageDraw

# take example from dataset to visualize output
pdf = processed_ds.limit(5).to_pandas()
row_n = 2
image = pdf.iloc[row_n]["image"]
annotations = pdf.iloc[row_n]["objects"]
width = pdf.iloc[row_n]["width"]
height = pdf.iloc[row_n]['height']

# Now you can use ImageDraw
draw = ImageDraw.Draw(image)

for i in range(len(annotations["bbox_id"])):
    box = annotations["bbox"][i]
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    # Check if coordinates are normalized or not
    if max(box) > 1.0:
        # Coordinates are un-normalized, no need to re-scale them
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
    else:
        # Coordinates are normalized, re-scale them
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

image

# COMMAND ----------

# MAGIC %md ### Preprocess data

# COMMAND ----------

import numpy as np
from PIL import Image
import io
import albumentations as A
from transformers import AutoImageProcessor
import ray

MODEL_NAME = "facebook/detr-resnet-50"

# Initialize the image processor
image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    do_pad=True
)

# Define the transformations
train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_area=25, clip=True),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False, verbose=True):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # Ensure bboxes is a list of lists
        bboxes = objects['bbox'].tolist() if isinstance(objects['bbox'], np.ndarray) else objects['bbox']

        
        # Ensure categories is a list
        categories = objects['category'].tolist() if isinstance(objects['category'], np.ndarray) else objects['category']

        # Ensure areas is a list
        areas = objects['area'].tolist() if isinstance(objects['area'], np.ndarray) else objects['area']

        if verbose:
          print(f"Image shape: {image.shape}")
          print(f"Bboxes: {objects['bbox']}")
          print(f"Bboxes post-transform: {bboxes}")
          print(f"Categories: {objects['category']}")
          print(f"Categories post-transform: {categories}")
          print(f"Areas: {objects['area']}")
          print(f"Areas post-transform: {areas}")

        # apply augmentations
        try:
            output = transform(image=image, bboxes=bboxes, category=categories)
            images.append(output["image"])

            # format annotations in COCO format
            formatted_annotations = format_image_annotations_as_coco(
                image_id, output["category"], areas, output["bboxes"]
            )
            annotations.append(formatted_annotations)
        except Exception as e:
            print(f"Error in transform: {e}")
            print(f"Image shape: {image.shape}")
            print(f"Bboxes: {bboxes}")
            print(f"Categories: {categories}")
            continue

    if not images:
        raise ValueError("No images were successfully processed")

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result
  
from functools import partial

# Define your partial functions
train_transform_batch = partial(
    augment_and_transform_batch, 
    transform=train_augment_and_transform, 
    image_processor=image_processor,
    verbose=True
)

validation_transform_batch = partial(
    augment_and_transform_batch, 
    transform=validation_transform, 
    image_processor=image_processor,
    verbose=True
)

# Apply to your processed datasets
processed_train_ds, processed_val_ds = processed_ds.train_test_split(test_size=0.2)
processed_train_ds = processed_train_ds.map_batches(
    train_transform_batch,
    batch_format="pandas",
    batch_size=100,
    concurrency=3,
    num_gpus=4
)

processed_val_ds = processed_val_ds.map_batches(
    validation_transform_batch,
    batch_format="pandas",
    batch_size=100,
    concurrency=3,
    num_gpus=4
)

# print(f"Train dataset: {processed_train_ds.take(10)}")
# print(f"Validation dataset: {processed_val_ds.take(10)}")

# COMMAND ----------

# MAGIC %md ### Preparing function to compute mAP

# COMMAND ----------

from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

# COMMAND ----------

from functools import partial
import numpy as np
from dataclasses import dataclass
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)

# COMMAND ----------

# MAGIC %md ### Persist training data to parquet (TO DO: convert to delta)

# COMMAND ----------

# MAGIC %md ### Train detection model

# COMMAND ----------

import torch
from transformers import Trainer
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import BatchFeature

import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func():

    # def collate_fn(batch):
    #     data = {}
    #     data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    #     data["labels"] = [x["labels"] for x in batch]
    #     if "pixel_mask" in batch[0]:
    #         data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    #     return data
    def collate_fn(batch):

        # Ray Data passes a dictionary of column names to arrays
        pixel_values = torch.stack([torch.as_tensor(item) for item in batch['pixel_values']])
        
        # Labels might be a list of dictionaries or a structured array
        labels = batch['labels']
        # labels = [x['labels'] for x in batch]
        # labels = [torch.as_tensor(x["labels"]) for x in batch]
        
        data = {
            "pixel_values": pixel_values,
            "labels": labels
        }
        
        # Add pixel_mask if it exists
        if 'pixel_mask' in batch:
            data["pixel_mask"] = torch.stack([torch.as_tensor(item) for item in batch['pixel_mask']])

        return data

    # Access Ray datsets in your train_func via ``get_dataset_shard``.
    # Ray Data shards all datasets across workers by default.
    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("validation")

    # # Debug: Print the structure of the first few batches from each dataset
    # print("\nDebug: First few batches in train dataset:")
    # for i, batch in enumerate(train_ds.iter_batches(batch_size=16)):
    #     print(f"Batch {i}:")
    #     for key, value in batch.items():
    #         print(f"  {key}: type={type(value)}, ", end="")
    #         if isinstance(value, np.ndarray):
    #             print(f"shape={value.shape}, dtype={value.dtype}")
    #         elif isinstance(value, dict):
    #             print("keys=" + ", ".join(value.keys()))
    #         else:
    #             print(f"value={value}")
    #     if i >= 2:  # Only print the first 3 batches
    #         break

    # print("\nDebug: First few batches in eval dataset:")
    # for i, batch in enumerate(eval_ds.iter_batches(batch_size=16)):
    #     print(f"Batch {i}:")
    #     for key, value in batch.items():
    #         print(f"  {key}: type={type(value)}, ", end="")
    #         if isinstance(value, np.ndarray):
    #             print(f"shape={value.shape}, dtype={value.dtype}")
    #         elif isinstance(value, dict):
    #             print("keys=" + ", ".join(value.keys()))
    #         else:
    #             print(f"value={value}")
    #     if i >= 2:  # Only print the first 3 batches
    #         break

    # Create Ray dataset iterables via ``iter_torch_batches``.
    train_iterable_ds = train_ds.iter_torch_batches(batch_size=8,
                                                    collate_fn=collate_fn
    )
    eval_iterable_ds = eval_ds.iter_torch_batches(batch_size=8,
                                                  collate_fn=collate_fn
    )

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
      
    training_args = TrainingArguments(
        output_dir="detr_finetuned_coco",
        max_steps=10,
        num_train_epochs=1,
        fp16=False,
        # per_device_train_batch_size=8,
        # dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_iterable_ds,
        eval_dataset=eval_iterable_ds,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    # Prepare your Transformers Trainer
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()

trainer = TorchTrainer(
    train_func,
    # You can pass in multiple datasets to the Trainer.
    datasets={"train": processed_train_ds, "validation": processed_val_ds},
    scaling_config=ScalingConfig(num_workers=2, 
                                 use_gpu=True, 
                                 resources_per_worker={"GPU": 4}
                                 )
)
trainer.fit()

# COMMAND ----------

from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments

import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="detr_finetuned_cppe5",
    num_train_epochs=30,
    fp16=False,
    per_device_train_batch_size=8,
    dataloader_num_workers=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cppe5["train"],
    eval_dataset=cppe5["validation"],
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()
