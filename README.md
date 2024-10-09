# Deep Learning on Databricks

This repository contains examples and best practices for training deep learning models on Databricks using frameworks such as Ray, Mosaic Composer, PyTorch, and TorchDistributor. It covers both single-node and distributed training approaches.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Frameworks](#frameworks)
- [Single-Node Training](#single-node-training)
- [Distributed Training](#distributed-training)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates how to leverage Databricks for efficient deep learning model training. It showcases various frameworks and techniques to optimize your workflow, whether you're working on a single node or distributing your training across a cluster.

## Prerequisites

- Databricks account
- Databricks Runtime for Machine Learning
- Basic knowledge of Python and deep learning concepts

## Installation

1. Clone this repository to your Databricks workspace.

## Usage

Each notebook in this repository demonstrates a different aspect of deep learning on Databricks. To get started:

1. Open the desired notebook in your Databricks workspace.
2. Attach it to a cluster running Databricks Runtime for Machine Learning.
3. Run the cells sequentially to understand the process and see the results.

## Frameworks

We cover the following frameworks:

- Ray
- Mosaic Composer
- PyTorch
- TorchDistributor

Each framework has its own dedicated notebook with examples and explanations.

## Single-Node Training

For single-node training, we recommend using a Databricks cluster with GPU support. Check the `single_node_training.ipynb` notebook for detailed examples.

## Distributed Training

Distributed training is crucial for large datasets and complex models. Example notebooks cover:

- TorchDistributor
- Mosaic Streaming
- Mosaic Composer
- Ray Train and Ray Data

## Best Practices

- Start simple with single-node training and then move to distributed training if required for use case
- Leverage MLflow for experiment tracking
- Optimize data loading with Mosaic Streaming or Ray Data

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

