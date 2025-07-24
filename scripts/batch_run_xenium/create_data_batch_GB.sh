#!/bin/bash

# Configuration - can be modified or overridden with command line arguments
PROJECT_DIR=${1:-"/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/20230907-GB-Xenium-Vis-sn/Raw/Neuronal_Panel/20240822_GB_CytAssist_Run03"} # folder including the raw xenium data folders
SCRNASEQ_FILE=${2:-"/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/20230907-GB-Xenium-Vis-sn/sn/GBmap_core.h5ad"} # the scRNAseq atlas
CELLTYPE_COLUMN=${5:-"cell_type"} # column pointing to the cell type annotation
OUTPUT_DIR=${3:-"logs"} # where to save the logs
SEGGER_DATA_DIR=${4:-"data_tidy/pyg_datasets/Neuronal_Panel"} # where to save intermediate segge files (graphs and embeddings)


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