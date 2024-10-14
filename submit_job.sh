#!/bin/bash

# To acquire the Singularity image, run:
# singularity pull docker://danielunyi42/segger_dev

# Pipeline 1: Data Preprocessing Parameters
OUTPUT_LOG_PREPROCESS="preprocess_output.log"  # Path to the output log file for data preprocessing
BASE_DIR="data_xenium"  # Base directory for input data
DATA_DIR="data_segger"  # Directory for output data
SAMPLE_TYPE="xenium"  # Type of sample being processed
TILE_WIDTH=120  # Width of each data tile
TILE_HEIGHT=120  # Height of each data tile
N_WORKERS_PREPROCESS=16  # Number of workers for parallel processing
RAM_PREPROCESS="16G"  # Total memory requested for the job

# Pipeline 2: Training Parameters
OUTPUT_LOG_TRAIN="train_output.log"  # Path to the output log file for training
DATASET_DIR="data_segger"  # Directory for dataset
MODELS_DIR="model_dir"  # Directory to save models
SAMPLE_TAG="first_training"  # Tag for the training sample
N_WORKERS_TRAIN=16  # Number of CPUs to request
RAM_TRAIN="16G"  # Amount of memory to request
GPUS=8  # Number of GPUs to request
GPU_MEM_TRAIN="8G"  # Amount of memory per GPU

# Pipeline 3: Prediction Parameters
OUTPUT_LOG_PREDICT="predict_output.log"  # Path to the output log file for prediction
SEGGER_DATA_DIR="data_segger"  # Directory containing the segger data
MODELS_DIR="model_dir"  # Directory containing the trained models
BENCHMARKS_DIR="benchmark_dir"  # Directory for saving the benchmark results
TRANSCRIPTS_FILE="data_xenium"  # Path to the transcripts file
KNN_METHOD="cuda"  # Method for KNN search
N_WORKERS_PREDICT=16  # Number of CPUs to request
RAM_PREDICT="16G"  # Amount of memory to request
GPU_MEM_PREDICT="8G"  # Amount of memory for GPU

# Paths and common variables
LOCAL_REPO_DIR="/omics/groups/OE0540/internal_temp/users/danielu/segger_dev"  # Where the segger_dev repository is located on the local machine
CONTAINER_DIR="/workspace/segger_dev"  # Where the segger_dev repository is located in the container
SINGULARITY_IMAGE="segger_dev_latest.sif"  # Path to the Singularity image

# Functions to run different pipelines
run_data_processing() {
    bsub -o "$OUTPUT_LOG_PREPROCESS" -n "$N_WORKERS_PREPROCESS" -R "rusage[mem=$RAM_PREPROCESS]" -q long \
    "singularity exec --bind $LOCAL_REPO_DIR:$CONTAINER_DIR \
    $SINGULARITY_IMAGE python3 src/segger/cli/create_dataset_fast.py \
    --base_dir '$BASE_DIR' \
    --data_dir '$DATA_DIR' \
    --sample_type '$SAMPLE_TYPE' \
    --tile_width $TILE_WIDTH \
    --tile_height $TILE_HEIGHT \
    --n_workers $N_WORKERS_PREPROCESS"
}

run_training() {
    bsub -o "$OUTPUT_LOG_TRAIN" -n "$N_WORKERS_TRAIN" -R "rusage[mem=$RAM_TRAIN]" -R "tensorcore" -gpu "num=$GPUS:j_exclusive=no:gmem=$GPU_MEM_TRAIN" -q gpu \
    "singularity exec --nv --bind $LOCAL_REPO_DIR:$CONTAINER_DIR \
    $SINGULARITY_IMAGE python3 src/segger/cli/train_model.py \
    --dataset_dir '$DATASET_DIR' \
    --models_dir '$MODELS_DIR' \
    --sample_tag '$SAMPLE_TAG' \
    --num_workers $N_WORKERS_TRAIN \
    --devices $GPUS"
}

run_prediction() {
    bsub -o "$OUTPUT_LOG_PREDICT" -n "$N_WORKERS_PREDICT" -R "rusage[mem=$RAM_PREDICT]" -R "tensorcore" -gpu "num=1:j_exclusive=no:gmem=$GPU_MEM_PREDICT" -q gpu \
    "singularity exec --nv --bind $LOCAL_REPO_DIR:$CONTAINER_DIR \
    $SINGULARITY_IMAGE python3 src/segger/cli/predict.py \
    --segger_data_dir '$SEGGER_DATA_DIR' \
    --models_dir '$MODELS_DIR' \
    --benchmarks_dir '$BENCHMARKS_DIR' \
    --transcripts_file '$TRANSCRIPTS_FILE' \
    --knn_method '$KNN_METHOD' \
    --num_workers $N_WORKERS_PREDICT"
}

# Main script logic
echo "Which pipelines would you like to run? (1: Data Processing, 2: Training, 3: Prediction)"
echo "Enter the pipeline numbers you want to run (e.g., '1 2 3' for all, or '1' for only data processing):"
read -r pipelines

for pipeline in $pipelines; do
    case $pipeline in
        1)
            echo "Running Data Processing..."
            run_data_processing
            ;;
        2)
            echo "Running Training..."
            run_training
            ;;
        3)
            echo "Running Prediction..."
            run_prediction
            ;;
        *)
            echo "Invalid choice: $pipeline"
            ;;
    esac
done
