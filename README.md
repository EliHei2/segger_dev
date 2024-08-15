# Segger

Segger is a cell segmentation model designed for single-molecule resolved datasets, leveraging the co-occurrence of nucleic and cytoplasmic molecules (e.g., transcripts). It employs a heterogeneous graph structure on molecules and nuclei, integrating fixed-radius nearest neighbor graphs for nuclei and molecules, along with edges connecting transcripts to nuclei based on spatial proximity.


![Segger Model](docs/images/Segger_model_08_2024.png)

## Table of Contents

- [Segger](#segger)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Download Pancreas Dataset](#download-pancreas-dataset)
  - [Creating Dataset](#creating-dataset)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Benchmarking](#benchmarking)
  - [Visualization](#visualization)
  - [License](#license)

## Introduction

Imaging-based spatial omics datasets present challenges in reliably segmenting single cells. Achieving accurate segmentation at single-cell resolution is crucial to unraveling multicellular mechanisms and understanding cell-cell communications in spatial omics studies. Despite the considerable progress and the variety of methods available for cell segmentation, challenges persist, including issues of over-segmentation, under-segmentation, and contamination from neighboring cells.

Here we introduce Segger, a cell segmentation model designed for single-molecule resolved datasets, leveraging the co-occurrence of nucleic and cytoplasmic molecules (e.g., transcripts). It employs a heterogeneous graph structure on molecules and nuclei, integrating fixed-radius nearest neighbor graphs for nuclei and molecules, along with edges connecting transcripts to nuclei based on spatial proximity. A heterogeneous graph neural network (GNN) is then used to propagate information across these edges to learn the association of molecules with nuclei. Post-training, the model refines cell borders by regrouping transcripts based on confidence levels, overcoming issues like nucleus-less cells or overlapping cells.

Benchmarks on 10X Xenium and MERSCOPE technologies reveal Segger's superiority in accuracy and efficiency over contemporary segmentation methods, such as Baysor, Cellpose, and simple nuclei-expansion. Segger can be pre-trained on one or more datasets and fine-tuned with new data, even acquired via different technologies. Its highly parallelizable nature allows for efficient training across multiple GPU machines, facilitated by recent graph learning techniques. Compared to other model-based methods like Baysor, JSTA, and Mesmer, Segger's training is orders of magnitude faster, making it ideal for integration into preprocessing pipelines for comprehensive spatial omics atlases.

## Installation

To install Segger, clone this repository and install the required dependencies:

```bash
git clone https://github.com/EliHei2/segger_dev.git
cd segger_dev
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate segger
```

## Download Pancreas Dataset

Download the Pancreas dataset from 10x Genomics:

1. Go to the [Xenium Human Pancreatic Dataset Explorer](https://www.10xgenomics.com/products/xenium-human-pancreatic-dataset-explorer).
2. Download the `transcripts.csv.gz` and `nucleus_boundaries.csv.gz` files.
3. Place these files in a directory, e.g., `data_raw/pancreas`.

## Dataset Creation

To create a dataset for the Segger model, use the `segger create_dataset` command. This command takes several arguments to customize the dataset creation process.

### Arguments

- `--data_dir`: Directory to store raw and processed data.
- `--xenium_dir`: Path to Xenium dataset.
- `--min_qv`: Minimum quality value for filtering transcripts (default: 30).
- `--d_x`: Step size in x direction for tiles (default: 180).
- `--d_y`: Step size in y direction for tiles (default: 180).
- `--x_size`: Width of each tile (default: 200).
- `--y_size`: Height of each tile (default: 200).
- `--margin_x`: Margin in x direction (default: null).
- `--margin_y`: Margin in y direction (default: null).
- `--r_tx`: Radius for building the graph (default: 3).
- `--val_prob`: Probability of assigning a tile to the validation set (default: 0.1).
- `--test_prob`: Probability of assigning a tile to the test set (default: 0.1).
- `--k_nc`: Number of nearest neighbors for nuclei (default: 3).
- `--dist_nc`: Distance threshold for nuclei (default: 4).
- `--k_tx`: Number of nearest neighbors for transcripts (default: 3).
- `--dist_tx`: Distance threshold for transcripts (default: 1).
- `--compute_labels`: Whether to compute edge labels (default: true).
- `--sampling_rate`: Rate of sampling tiles (default: 1).
- `--workers`: Number of workers for parallel processing (default: 4).

### Example Usage

```bash
segger create_dataset \
--data_dir ./data_tidy/pyg_datasets/pancreas \
--xenium_dir ./data_raw/pancreas
```

## Training

To train the Segger model, use the `segger train` command. This command takes several arguments to customize the training process.

### Arguments

- `--data_dir`: Path to the data loader directory.
- `--model_dir`: Root directory for logs and model checkpoints.
- `--batch_size_train`: Batch size for training (default: 4).
- `--batch_size_val`: Batch size for validation (default: 4).
- `--init_emb`: Initial embedding size (default: 8).
- `--hidden_channels`: Number of hidden channels (default: 64).
- `--out_channels`: Number of output channels (default: 16).
- `--heads`: Number of attention heads (default: 4).
- `--aggr`: Aggregation method (default: "sum").
- `--accelerator`: Type of accelerator (default: "cuda").
- `--strategy`: Training strategy (default: "auto").
- `--precision`: Precision mode (default: "16-mixed").
- `--devices`: Number of devices (default: 4).
- `--epochs`: Number of epochs (default: 100).

### Example Usage

```bash
segger train \
--data_dir ./data_tidy/pyg_datasets/pancreas \
--model_dir ./models/pancreas
```

## Prediction

To make predictions using a trained Segger model, use the `segger predict` command. This command takes several arguments to customize the prediction process.

### Arguments

- `--checkpoint_path`: Path to the model checkpoint.
- `--dataset_path`: Path to the dataset directory.
- `--output_path`: Path to save the predictions.
- `--score_cut`: Score cut-off for predictions (default: 0.5).
- `--k_nc`: Number of nearest neighbors for nuclei (default: 5).
- `--dist_nc`: Distance threshold for nuclei (default: 3).
- `--k_tx`: Number of nearest neighbors for transcripts (default: 5).
- `--dist_tx`: Distance threshold for transcripts (default: 1).
- `--batch_size`: Size of batch used by data loader (default: 1).
- `--workers`: Workers used to load data (default: 0).

### Example Usage

```bash
segger predict \
--checkpoint_path ./models/pancreas/lightning_logs/version_0/checkpoints/epoch=99-step=100.ckpt \
--dataset_path ./data_tidy/pyg_datasets/pancreas
```

## Benchmarking

Benchmarking utilities are provided to evaluate the performance of the Segger model. You can find these utilities in the `benchmark` directory.

## Visualization

Visualization scripts are also provided to help visualize the results. You can find these scripts in the `benchmark` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
