# Training the `Segger` model

## Introduction

The training module makes use of **PyTorch Lightning** for efficient and scalable training, alongside **PyTorch Geometric** for processing the graph-based data. The module is built to handle multi-GPU setups and allows the flexibility to adjust hyperparameters, aggregation methods, and embedding sizes.

## Key Components

### 1. **SpatialTranscriptomicsDataset**

The `SpatialTranscriptomicsDataset` class is used to load and manage spatial transcriptomics data stored in the format of PyTorch Geometric `Data` objects. It inherits from `InMemoryDataset` to load preprocessed datasets, ensuring efficient in-memory data handling for training and validation phases.

- **Root Path**: The root directory contains the dataset, which is expected to have separate folders for training, validation, and test sets.
- **Raw and Processed Data**: The module expects datasets in the form of processed PyTorch files, and the dataset class is responsible for loading them efficiently.

### 2. **Segger Model**

The `Segger` model is a custom graph neural network designed to work with heterogeneous graph data. It takes both **transcript (tx)** and **boundary (bd)** nodes, utilizing attention mechanisms for better feature aggregation. Key parameters such as `num_tx_tokens`, `init_emb`, `hidden_channels`, `out_channels`, and `heads` allow the user to control the model's architecture and initial embedding sizes.

- **Heterogeneous Graph Support**: The model is converted to handle different node types using `to_hetero` from PyTorch Geometric. The transformation allows the model to handle multiple relations like `belongs` (tx to bd) and `neighbors` (tx to tx).

### 3. **LitSegger**

`LitSegger` is the PyTorch Lightning wrapper around the Segger model, which handles training, validation, and optimization. This wrapper facilitates the integration with Lightning’s trainer, allowing easy multi-GPU and distributed training.

### 4. **Training Pipeline**

The module provides an easily configurable pipeline for training the Segger model:

- **Datasets**: Training and validation datasets are loaded using `SpatialTranscriptomicsDataset` with paths provided via arguments.
- **DataLoader**: The `DataLoader` class from PyTorch Geometric handles batching and shuffling of the dataset for both training and validation.
- **Trainer**: PyTorch Lightning's `Trainer` class is used to manage the training loop, supporting GPU acceleration, mixed precision, and flexible strategies for distributed training.

## Usage and Configuration

### Command-Line Arguments

The module accepts various command-line arguments that allow for flexible configuration:

- `--train_dir`: Path to the training data directory. This directory should include `processed` and `raw` subdirectories. The direcotry `processed` should include the `pyg` `HeteroData` objects.
- `--val_dir`: Path to the validation data directory. This directory should include `processed` and `raw` subdirectories. The direcotry `processed` should include the `pyg` `HeteroData` objects.
- `--batch_size_train`: Batch size for training (default: 4).
- `--batch_size_val`: Batch size for validation (default: 4).
- `--num_tx_tokens`: Number of unique transcript tokens for embedding (default: 500).
- `--init_emb`: Initial embedding size (default: 8).
- `--hidden_channels`: Number of hidden channels in the model (default: 64).
- `--out_channels`: Number of output channels (default: 16).
- `--heads`: Number of attention heads (default: 4).
- `--mid_layers`: Number of middle layers in the model (default: 1).
- `--aggr`: Aggregation method for multi-node aggregation (default: sum).
- `--accelerator`: Type of accelerator for training, e.g., `cuda`.
- `--strategy`: Training strategy for distributed or multi-GPU setups (default: auto).
- `--precision`: Precision for training, e.g., `16-mixed` for mixed precision.
- `--devices`: Number of devices (GPUs) to use for training.
- `--epochs`: Number of training epochs.
- `--default_root_dir`: Directory where logs, checkpoints, and models will be saved.

### Example Training Command

The module can be executed from the command line as follows:

```bash
# this is for LSF clusters.
bsub -o logs -R "tensorcore" -gpu "num=4:gmem=20G" -q gpu CUDA_LAUNCH_BLOCKING=1 \
python path/to/train_model.py \
  --train_dir path/to/train/tiles\
  --val_dir path/to/val/tiles \
  --batch_size_train 4 \
  --batch_size_val 4 \
  --num_tx_tokens 500 \
  --init_emb 8 \
  --hidden_channels 64 \
  --out_channels 16 \
  --heads 4 \
  --mid_layers 1 \
  --aggr sum \
  --accelerator cuda \
  --strategy auto \
  --precision 16-mixed \
  --devices 4 \
  --epochs 100 \
  --default_root_dir ./models/clean2
```

This example submits a job to train the Segger model on four GPUs with a batch size of 4 for both training and validation, utilizing 16-bit mixed precision.
