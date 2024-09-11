#!/bin/bash

# Submit the job with bsub, requesting 4 GPUs and setting GPU memory limit to 20G
bsub -o logs_0910 -R "tensorcore" -gpu "num=4:gmem=20G" -q gpu CUDA_LAUNCH_BLOCKING=1 \
python ../segger_dev/scripts/train_model.py \
  --train_dir data_tidy/pyg_datasets/clean_parallel/train_tiles \
  --val_dir data_tidy/pyg_datasets/clean_parallel/val_tiles \
  --batch_size_train 4 \
  --batch_size_val 4 \
  --num_tx_tokens 1000 \
  --init_emb 8 \
  --hidden_channels 64 \
  --out_channels 16 \
  --heads 4 \
  --mid_layers 1 \
  --aggr sum \
  --accelerator cpu \
  --strategy auto \
  --precision 16-mixed \
  --devices 4 \
  --epochs 100 \
  --default_root_dir ./models/clean_parallel
