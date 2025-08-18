#!/bin/bash

# Set default directories
DATA_ROOT=${1:-"data_tidy/pyg_datasets/project24_MNG_final"}
MODEL_TAG=${2:-"output-XETG00423__0052506__mng_04_TMA__20250310__160549"}
MODEL_VERSION=${3:-0}
OUTPUT_DIR=${4:-"logs"}
MODELS_ROOT=${5:-"./models/project24_MNG_pqdm"}
BENCHMARKS_ROOT=${6:-"data_tidy/benchmarks/project24_MNG_final"}
TRANSCRIPTS_ROOT=${7:-"/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/domenico_temp/xenium/xenium_output_files"}
GPU_MEM=${8:-"39G"}
SYSTEM_MEM=${9:-"200GB"}
FORCE=${10:-false}

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get list of all samples in data root
ALL_SAMPLES=($(ls $DATA_ROOT))

# Process samples
processed=0
skipped=0
for SAMPLE in "${ALL_SAMPLES[@]}"; do
    # Skip model tag sample
    if [ "$SAMPLE" == "$MODEL_TAG" ]; then
        continue
    fi

    # Define output file path
    H5AD_FILE1="${BENCHMARKS_ROOT}/${SAMPLE}/${SAMPLE}_0.75_False_4_7.5_5_3_20250709/segger_adata.h5ad"
    H5AD_FILE2="${BENCHMARKS_ROOT}/${SAMPLE}/${SAMPLE}_0.75_False_4_7.5_5_3_20250714/segger_adata.h5ad"
    H5AD_FILE3="${BENCHMARKS_ROOT}/${SAMPLE}/${SAMPLE}_0.75_False_4_7.5_5_3_20250728/segger_adata.h5ad"

    # Check if processing should be skipped
    if [[ -f "$H5AD_FILE1" || -f "$H5AD_FILE2" || -f "$H5AD_FILE3" ]] && [ "$FORCE" != "true" ]; then
        echo "segger_adata.h5ad exists for $SAMPLE, skipping..."
        ((skipped++))
        continue
    fi

    echo "Submitting job for sample: $SAMPLE"
    ((processed++))
    
    # Create benchmark output directory
    mkdir -p "${BENCHMARKS_ROOT}/${SAMPLE}"
    
    # Submit with bsub
    bsub -o ${OUTPUT_DIR}/segmentation_${SAMPLE}.log \
         -e ${OUTPUT_DIR}/segmentation_${SAMPLE}.err \
         -gpu "num=1:j_exclusive=yes:gmem=${GPU_MEM}" \
         -m gpu-a100-40gb \
         -R "rusage[mem=${SYSTEM_MEM}]" \
         -q gpu-debian \
         python ../segger_dev/scripts/batch_run_xenium/predict_batch.py \
            --seg_tag "$MODEL_TAG" \
            --sample_id "$SAMPLE" \
            --model_version "$MODEL_VERSION" \
            --gpu_id "0" \
            --models_root "$MODELS_ROOT" \
            --data_root "$DATA_ROOT" \
            --output_root "$BENCHMARKS_ROOT" \
            --transcripts_root "$TRANSCRIPTS_ROOT" \
            --min_transcripts 5 \
            --score_cut 0.75 
done

echo "Job submission complete"
echo "Processed: $processed samples"
echo "Skipped: $skipped samples with existing segger_adata.h5ad"
echo "Configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Model tag: $MODEL_TAG"
echo "  Model version: $MODEL_VERSION"
echo "  Output dir: $OUTPUT_DIR"
echo "  Models root: $MODELS_ROOT"
echo "  Benchmarks root: $BENCHMARKS_ROOT"
echo "  Transcripts root: $TRANSCRIPTS_ROOT"
echo "  GPU memory: $GPU_MEM"
echo "  System memory: $SYSTEM_MEM"
echo "  Force reprocess: $FORCE"