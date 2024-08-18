#!/bin/bash

# Activate conda environment
source activate segger

# Run the Python script to train the model
python src/scripts/train_model.py \
  --train_dir data_tidy/pyg_datasets/pancreas/train_tiles/processed \
  --val_dir data_tidy/pyg_datasets/pancreas/val_tiles/processed \
  --batch_size_train 4 \
  --batch_size_val 4 \
  --init_emb 8 \
  --hidden_channels 64 \
  --out_channels 16 \
  --heads 4 \
  --aggr sum \
  --accelerator cuda \
  --strategy auto \
  --precision 16-mixed \
  --devices 4 \
  --epochs 100 \
  --default_root_dir ./models/pancreas
