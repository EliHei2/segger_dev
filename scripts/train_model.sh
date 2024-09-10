#!/bin/bash

# Submit the job with bsub, requesting 4 GPUs and setting GPU memory limit to 20G
bsub -o logs_0909 -R "tensorcore" -gpu "num=4:gmem=20G" -q gpu CUDA_LAUNCH_BLOCKING=1 \
<<EOF
# Activate conda environment inside the job
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust the path to where your conda is installed
conda activate segger_dev

# Run the Python script for training
python ../segger_dev/scripts/train_model.py \
  --train_dir data_tidy/pyg_datasets/clean2/train_tiles \
  --val_dir data_tidy/pyg_datasets/clean2/val_tiles \
  --batch_size_train 4 \
  --batch_size_val 4 \
  --num_tx_tokens 1000 \
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
EOF
