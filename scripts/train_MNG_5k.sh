
DATA_ROOT="data_tidy/pyg_datasets"
MODELS_ROOT="models"
PROJECT_NAME="MNG_5k"



for folder in "$DATA_ROOT"/*; do
    if [ -d "$folder" ]; then
        echo "Submitting job for $folder"
        bsub -o train_yiheng_5k \
             -gpu num=4:j_exclusive=yes:gmem=20.7G \
             -R "rusage[mem=100GB]" \
             -q gpu-debian \
             python /dkfz/cluster/gpu/data/OE0606/elihei/segger_dev/scripts/train_model.py --data_dir "$DATA_ROOT/$PROJECT_NAME/$folder" --model_dir "$MODELS_ROOT/$PROJECT_NAME/$folder"
    fi
done