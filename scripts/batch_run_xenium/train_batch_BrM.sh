DATA_ROOT="data_tidy/pyg_datasets" # this is the ../ parent folder of where segger tiles (graphs and embeddings) are saved
MODELS_ROOT="models" # this is where the trained models are stored
PROJECT_NAME="BrM" # this is the folder tag for the datasets DATA_ROOT/PROJECT_NAME and the folder in MODELS_ROOT where the models are gonna be saved

for full_path in "$DATA_ROOT/$PROJECT_NAME"/*; do
    if [ -d "$full_path" ]; then
        echo "$full_path"
        dataset_tag=$(basename "$full_path")
        echo "Submitting job for $dataset_tag"
        bsub -o logs/train_${dataset_tag}.log \
             -gpu num=1:gmem=30G \
             -R "rusage[mem=100GB]" \
             -q gpu \
             python ../segger_dev/scripts/batch_run_xenium/train_batch.py \
             --data_dir "$DATA_ROOT/$PROJECT_NAME/$dataset_tag" \
             --models_dir "$MODELS_ROOT/$PROJECT_NAME/$dataset_tag"
    fi
done