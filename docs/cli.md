## Segger Command Line Interface

### 1. Creating a Dataset

The `create_dataset` command helps you to build a dataset for spatial transcriptomics. Here’s a breakdown of the options available:

<!-- termynal -->
```console
// Example: Creating a dataset for spatial transcriptomics
python3 src/segger/cli/create_dataset_fast.py \
    --base_dir /path/to/raw_data \
    --data_dir /path/to/save/processed_data \
    --sample_type xenium \
    --scrnaseq_file /path/to/scrnaseq_file \
    --celltype_column celltype_column_name \
    --k_bd 3 \
    --dist_bd 15.0 \
    --k_tx 3 \
    --dist_tx 5.0 \
    --tile_width 200 \
    --tile_height 200 \
    --neg_sampling_ratio 5.0 \
    --frac 1.0 \
    --val_prob 0.1 \
    --test_prob 0.2 \
    --n_workers 16
```

#### Parameters

| Parameter            | Description                                                                             | Default Value |
|----------------------|-----------------------------------------------------------------------------------------|---------------|
| `base_dir`           | Directory containing the raw dataset (e.g., transcripts, boundaries).                   | -             |
| `data_dir`           | Directory to save the processed Segger dataset (in PyTorch Geometric format).           | -             |
| `sample_type`        | The sample type of the raw data, e.g., "xenium" or "merscope".                          | None          |
| `scrnaseq_file`      | Path to the scRNAseq file.                                                              | None          |
| `celltype_column`    | Column name for cell type annotations in the scRNAseq file.                             | None          |
| `k_bd`               | Number of nearest neighbors for boundary nodes.                                         | `3`           |
| `dist_bd`            | Maximum distance for boundary neighbors.                                                | `15.0`        |
| `k_tx`               | Number of nearest neighbors for transcript nodes.                                       | `3`           |
| `dist_tx`            | Maximum distance for transcript neighbors.                                              | `5.0`         |
| `tile_width`         | Width of the tiles in pixels (ignored if `tile_size` is provided).                      | None          |
| `tile_height`        | Height of the tiles in pixels (ignored if `tile_size` is provided).                     | None          |
| `neg_sampling_ratio` | Ratio of negative samples.                                                              | `5.0`         |
| `frac`               | Fraction of the dataset to process. Useful for subsampling large datasets.              | `1.0`         |
| `val_prob`           | Proportion of the dataset used for validation split.                                    | `0.1`         |
| `test_prob`          | Proportion of the dataset used for testing split.                                       | `0.2`         |
| `n_workers`          | Number of workers for parallel processing.                                              | `1`           |


#### Key Updates
- **Faster Dataset Creation** This method is way faster due to the use of ND-tree-based partitioning and parallel processing.

!!! note "Customizing Your Dataset"
    - **dataset_type**: Defines the type of spatial transcriptomics data. Currently, **xenium** and **merscope** are supported and have been tested.
    - **val_prob, test_prob**: Control the dataset portions for validation and testing. Adjust based on your dataset size and evaluation needs.
    - **frac**: Specifies the fraction of the dataset to process. Reducing `frac` can be useful when working with very large datasets, allowing for faster dataset creation by only processing a subset of the data.

!!! tip "Faster Dataset Creation"
    Increasing the number of workers (`n_workers`) can significantly accelerate the dataset creation process, especially for large datasets, by taking advantage of parallel processing across multiple CPU cores.

!!! tip "Enhancing Segmentation Accuracy with scRNA-seq"
    Incorporating single cell RNA sequencing (scRNA-seq) data as features can provide additional biological context, improving the accuracy of the segger model.
---

### 2. Training a Model

The `train` command initializes and trains a model using the dataset created. Here are the key parameters:

<!-- termynal -->
```console
// Example: Training a segger model
$ python3 src/segger/cli/train_model.py \
    --dataset_dir /path/to/saved/processed_data \
    --models_dir /path/to/save/model/checkpoints \
    --sample_tag first_training \
    --init_emb 8 \
    --hidden_channels 32 \
    --num_tx_tokens 500 \
    --out_channels 8 \
    --heads 2 \
    --num_mid_layers 2 \
    --batch_size 4 \
    --num_workers 2 \
    --accelerator cuda \
    --max_epochs 200 \
    --devices 4 \
    --strategy auto \
    --precision 16-mixed
```

#### Parameters

| Parameter          | Description                                                                             | Default Value |
|--------------------|-----------------------------------------------------------------------------------------|---------------|
| `dataset_dir`      | Directory containing the processed Segger dataset (in PyTorch Geometric format).        | -             |
| `models_dir`       | Directory to save the trained model and training logs.                                  | -             |
| `sample_tag`       | Tag used to identify the dataset during training.                                       | -             |
| `init_emb`         | Size of the embedding layer for input data.                                             | `8`           |
| `hidden_channels`  | Number of hidden units in each layer of the neural network.                             | `32`          |
| `num_tx_tokens`    | Number of transcript tokens used during training.                                       | `500`         |
| `out_channels`     | Number of output channels from the model.                                               | `8`           |
| `heads`            | Number of attention heads used in graph attention layers.                               | `2`           |
| `num_mid_layers`   | Number of mid layers in the model.                                                      | `2`           |
| `batch_size`       | Number of samples to process per training batch.                                        | `4`           |
| `num_workers`      | Number of workers to use for parallel data loading.                                     | `2`           |
| `accelerator`      | Device used for training (e.g., `cuda` for GPU or `cpu`).                               | `cuda`        |
| `max_epochs`       | Number of training epochs.                                                              | `200`         |
| `devices`          | Number of devices (GPUs) to use during training.                                        | `4`           |
| `strategy`         | Strategy used for training (e.g., `ddp` for distributed training or `auto`).            | `auto`        |
| `precision`        | Precision used for training (e.g., `16-mixed` for mixed precision training).            | `16-mixed`    |

!!! tip "Optimizing training time"
    - **devices**: Use multiple GPUs by increasing the `devices` parameter to further accelerate training.
    - **batch_size**: A larger batch size can speed up training, but requires more memory. Adjust based on your hardware capabilities.
    - **epochs**: Increasing the number of epochs can improve model performance by allowing more learning cycles, but it will also extend the overall training time. Balance this based on your time constraints and hardware capacity.

!!! warning "Ensure Correct CUDA and PyTorch Setup"
    Before using the `--accelerator cuda` flag, ensure your system has CUDA installed and configured correctly. Also, check that the installed CUDA version is compatible with your PyTorch and PyTorch Geometric versions.

---

### 3. Making Predictions

After training the model, use the `predict` command to make predictions on new data:

<!-- termynal -->
```console
// Example: Make predictions using a trained model
$ python3 src/segger/cli/predict_fast.py \
    --segger_data_dir /path/to/saved/processed_data \
    --models_dir /path/to/saved/model/checkpoints \
    --benchmarks_dir /path/to/save/segmentation/results \
    --transcripts_file /path/to/raw_data/transcripts.parquet \
    --batch_size 1 \
    --num_workers 1 \
    --model_version 0 \
    --save_tag segger_embedding_1001 \
    --min_transcripts 5 \
    --cell_id_col segger_cell_id \
    --use_cc false \
    --knn_method cuda \
    --file_format anndata \
    --k_bd 4 \
    --dist_bd 12.0 \
    --k_tx 5 \
    --dist_tx 5.0
```

#### Parameters

| Parameter             | Description                                                                              | Default Value |
|-----------------------|------------------------------------------------------------------------------------------|---------------|
| `segger_data_dir`     | Directory containing the processed Segger dataset (in PyTorch Geometric format).        | -             |
| `models_dir`          | Directory containing the trained models.                                                | -             |
| `benchmarks_dir`      | Directory to save the segmentation results, including cell boundaries and associations. | -             |
| `transcripts_file`    | Path to the transcripts.parquet file.                                                   | -             |
| `batch_size`          | Number of samples to process per batch during prediction.                               | `1`           |
| `num_workers`         | Number of workers for parallel data loading.                                            | `1`           |
| `model_version`       | Model version number to load for predictions, corresponding to the version from training logs. | `0`           |
| `save_tag`            | Tag used to name and organize the segmentation results.                                 | `segger_embedding_1001` |
| `min_transcripts`     | Minimum number of transcripts required for segmentation.                                | `5`           |
| `cell_id_col`         | Column name for cell IDs in the output data.                                            | `segger_cell_id` |
| `use_cc`              | Whether to use connected components for grouping transcripts without direct nucleus association. | `False`       |
| `knn_method`          | Method for KNN computation (e.g., `cuda` for GPU-based computation).                    | `cuda`        |
| `file_format`         | Format for the output segmentation data (e.g., `anndata`).                              | `anndata`     |
| `k_bd`                | Number of nearest neighbors for boundary nodes.                                         | `4`           |
| `dist_bd`             | Maximum distance for boundary nodes.                                                    | `12.0`        |
| `k_tx`                | Number of nearest neighbors for transcript nodes.                                       | `5`           |
| `dist_tx`             | Maximum distance for transcript nodes.                                                  | `5.0`         |

!!! tip "Improving Prediction Pipeline"
    - **batch_size**: A larger batch size can speed up training, but requires more memory. Adjust based on your hardware capabilities.
    - **use_cc**: Enabling connected component analysis can improve the accuracy of transcript assignments.

!!! warning "Ensure Correct CUDA, cuVS, and PyTorch Setup"
    Before using the `knn_method cuda` flag, ensure your system has CUDA installed and configured properly. Also, verify that the installed CUDA version is compatible with your cuPy, cuVS, PyTorch, and PyTorch Geometric versions.

---

### 4. Running the Entire Pipeline

The `submit_job.py` script allows you to run the complete Segger pipeline or specific stages like dataset creation, training, or prediction. The pipeline execution is determined by the configuration provided in a YAML file, supporting various environments like Docker, Singularity, and HPC systems (with LSF, Slurm support is planned).

#### Selecting Pipelines
You can run the three stages—dataset creation, training, and prediction—sequentially or independently by specifying the pipelines in the YAML configuration file:

    - `1` for dataset creation
    - `2` for model training
    - `3` for prediction

This allows you to run the full pipeline or just specific steps. Set the desired stages under the pipelines field in your YAML file.

#### Running the Pipeline

Use the following command to run the pipeline:

```console
python3 submit_job.py --config_file=filename.yaml
```

- If no `--config_file` is provided, the default `config.yaml` file will be used.

### 5. Containerization

For users who want a portable, containerized environment, segger supports both Docker and Singularity containers. These containers provide a consistent runtime environment with all dependencies pre-installed.

#### Using Docker

You can pull the segger Docker image from Docker Hub with this command:

```console
docker pull danielunyi42/segger_dev:cuda121
```

To run the pipeline in Docker, make sure your YAML configuration includes the following settings:

- `use_singularity`: false
- `use_lsf`: false

Afterwards, run the pipeline inside the Docker container with the same `submit_job.py` command.

#### Using Singularity
For a Singularity environment, pull the image with:

```console
singularity pull docker://danielunyi42/segger_dev:cuda121
```

Ensure `use_singularity: true` in the YAML file and specify the Singularity image file (e.g., `segger_dev_latest.sif`) in the `singularity_image` field.

!!! note "Containerization"
    - The segger Docker image currently supports CUDA 11.8 and CUDA 12.1.

### 6. HPC Environments

Segger also supports HPC environments with LSF job scheduling. To run the pipeline on an HPC cluster using LSF, set `use_lsf: true` in your YAML configuration.

If your HPC system supports Slurm, a similar setup is planned and will be introduced soon.
