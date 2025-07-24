#!/bin/bash

# Configuration - can be modified or overridden with command line arguments
PROJECT_DIR=${1:-"/omics/odcf/analysis/OE0606_projects/xenium_projects/temp/20250212_Xenium5k/20250212__133739__051_B450_Run01_2025-02-12/"} # folder including the raw xenium data folders
SCRNASEQ_FILE=${2:-"/omics/groups/OE0606/internal/amathiou/Projects/202403_BrainMets/snRNAseq/merged.Annotation_merged.h5ad"} # the scRNAseq atlas
CELLTYPE_COLUMN=${5:-"Annotation_merged"} # column pointing to the cell type annotation
OUTPUT_DIR=${3:-"logs"} # where to save the logs
SEGGER_DATA_DIR=${4:-"data_tidy/pyg_datasets/BrM"} # where to save intermediate segge files (graphs and embeddings)


# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p $SEGGER_DATA_DIR

# Get list of samples to process (only XETG samples)
SAMPLES=($(ls $PROJECT_DIR | grep "^output-XETG"))

# Submit jobs for each sample
for SAMPLE in "${SAMPLES[@]}"; do
    echo "Submitting job for sample: $SAMPLE"
    
    # Create output directory for the sample
    mkdir -p "${SEGGER_DATA_DIR}/${SAMPLE}"
    
    # Submit with bsub
    bsub -o ${OUTPUT_DIR}/preprocess_${SAMPLE}.log \
         -e ${OUTPUT_DIR}/preprocess_${SAMPLE}.err \
         -R "rusage[mem=300GB]" \
         -n 5 \
         -q highmem-debian \
         python ../segger_dev/scripts/batch_run_xenium/create_data_batch.py \
            --sample_id "$SAMPLE" \
            --project_dir "$PROJECT_DIR" \
            --scrna_file "$SCRNASEQ_FILE" \
            --output_dir "$SEGGER_DATA_DIR" \
            --celltype_column "$CELLTYPE_COLUMN" \
            --n_workers 5 \
            --k_tx 5 \
            --dist_tx 5.0 \
            --subsample_frac 0.1
done

echo "All preprocessing jobs submitted"