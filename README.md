# Segger

![Segger Model](docs/images/Segger_model_08_2024.png)

Segger is a cell segmentation model designed for single-molecule resolved datasets, leveraging the co-occurrence of nucleic and cytoplasmic molecules (e.g., transcripts). It employs a heterogeneous graph structure on molecules and nuclei, integrating fixed-radius nearest neighbor graphs for nuclei and molecules, along with edges connecting transcripts to nuclei based on spatial proximity.

## Table of Contents

- [Segger](#segger)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Download Pancreas Dataset](#download-pancreas-dataset)
  - [Creating Dataset](#creating-dataset)
    - [Example Usage](#example-usage)
  - [Training](#training)
    - [Example Usage](#example-usage-1)
  - [Prediction](#prediction)
    - [Example Usage](#example-usage-2)
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

## Creating Dataset

To create a dataset for Segger, use the `create_data.py` script. The script takes several arguments to customize the dataset creation process.

### Example Usage

```bash
python create_data.py --transcripts_path data_raw/pancreas/transcripts.csv.gz --nuclei_path data_raw/pancreas/nucleus_boundaries.csv.gz --output_dir data_tidy/pyg_datasets/pancreas --d_x 180 --d_y 180 --x_size 200 --y_size 200 --r 3 --val_prob 0.1 --test_prob 0.1 --k_nc 3 --dist_nc 10 --k_tx 5 --dist_tx 3 --compute_labels True --sampling_rate 1
```

This command will process the Pancreas dataset and save the processed data in the specified output directory.

## Training

To train the Segger model, use the `train.py` script. The script takes several arguments to customize the training process.

### Example Usage

```bash
python train.py --train_dir data_tidy/pyg_datasets/pancreas/train_tiles/processed --val_dir data_tidy/pyg_datasets/pancreas/val_tiles/processed --test_dir data_tidy/pyg_datasets/pancreas/test_tiles/processed --epochs 100 --batch_size_train 4 --batch_size_val 4 --learning_rate 1e-3 --init_emb 8 --hidden_channels 64 --out_channels 16 --heads 4 --aggr sum --accelerator cuda --strategy auto --precision 16-mixed --devices 4 --default_root_dir ./models/pancreas
```

This command will train the Segger model on the processed Pancreas dataset and save the trained model in the specified output directory.

## Prediction

To make predictions using a trained Segger model, use the `predict.py` script. The script takes several arguments to customize the prediction process.

### Example Usage

```bash
python predict.py --train_dir data_tidy/pyg_datasets/pancreas/train_tiles/processed --val_dir data_tidy/pyg_datasets/pancreas/val_tiles/processed --test_dir data_tidy/pyg_datasets/pancreas/test_tiles/processed --checkpoint_path ./models/pancreas/lightning_logs/version_0/checkpoints/epoch=99-step=100.ckpt --batch_size 1 --init_emb 8 --hidden_channels 64 --out_channels 16 --heads 4 --aggr sum --accelerator cuda --devices 1 --default_root_dir ./log_final --score_cut 0.5 --k_nc 4 --dist_nc 20 --k_tx 5 --dist_tx 10
```

This command will use the trained Segger model to make predictions on the Pancreas dataset and save the predictions in the specified output directory.

## Benchmarking

Benchmarking utilities are provided to evaluate the performance of the Segger model. You can find these utilities in the `benchmark` directory.

## Visualization

Visualization scripts are also provided to help visualize the results. You can find these scripts in the `benchmark` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
