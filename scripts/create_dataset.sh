#!/bin/bash

# Ensure the script exits if a command fails
set -e

# Define the data directory paths
RAW_DATA_DIR="data_raw/pancreatic"
PROCESSED_DATA_DIR="data_tidy/pyg_datasets/pancreatic"

# Define the data URLs
TRANSCRIPTS_URL="https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/transcripts.csv.gz"
NUCLEI_URL="https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/nucleus_boundaries.csv.gz"

# Run the data preparation script
python scripts/create_dataset.py \
    --raw_data_dir $RAW_DATA_DIR \
    --processed_data_dir $PROCESSED_DATA_DIR \
    --transcripts_url $TRANSCRIPTS_URL \
    --nuclei_url $NUCLEI_URL \
    --min_qv 30 \
    --d_x 180 \
    --d_y 180 \
    --x_size 200 \
    --y_size 200 \
    --r_tx 3 \
    --val_prob 0.1 \
    --test_prob 0.1 \
    --k_nc 3 \
    --dist_nc 10 \
    --k_tx 5 \
    --dist_tx 3 \
    --compute_labels True \
    --sampling_rate 0.1 \
    --parallel \
    --num_workers 4


git config --global user.email elyas.heidari@dkfz-heidelberg.de