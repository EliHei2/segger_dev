#!/bin/bash

# Directory for processed data

LOG_DIR="logs_cli"
SAMPLE_TAG="Xenium_FFPE_Human_Breast_Cancer_Rep1"
DATASET_DIR="data_raw/xenium"
TRANSCRIPTS_FILE="${SAMPLE_TAG}/transcripts.parquet"
BOUNDARIES_FILE="${SAMPLE_TAG}/nucleus_boundaries.parquet"
DATA_DIR="data_tidy/pyg_datasets/${SAMPLE_TAG}"
GRID_SIZE=10  # 10x10 grid
OVERLAP=50  # Overlap between tiles (in the same units as your coordinates)

# Create logs directory if not exists
mkdir -p $LOG_DIR

# Load the dataset to calculate the bounds
python -c "
from segger.data.io import XeniumSample
sample = XeniumSample()
sample.load_transcripts(base_path='$DATASET_DIR', sample='$SAMPLE_TAG', transcripts_filename='$TRANSCRIPTS_FILE', file_format='parquet')
sample.load_boundaries('$DATASET_DIR/$BOUNDARIES_FILE', file_format='parquet')
x_min, y_min, x_max, y_max = sample.x_min, sample.y_min, sample.x_max, sample.y_max
tile_width = (x_max - x_min) / $GRID_SIZE + $OVERLAP
tile_height = (y_max - y_min) / $GRID_SIZE + $OVERLAP
print(f'{x_min},{y_min},{x_max},{y_max},{tile_width},{tile_height}')
" > bounds.txt

# Read the calculated bounds
read X_MIN_GLOBAL Y_MIN_GLOBAL X_MAX_GLOBAL Y_MAX_GLOBAL TILE_WIDTH TILE_HEIGHT < <(cat bounds.txt)

# Iterate over a GRID_SIZE x GRID_SIZE grid
for i in $(seq 0 $(($GRID_SIZE - 1)))
do
    for j in $(seq 0 $(($GRID_SIZE - 1)))
    do
        # Calculate the bounding box for this tile
        X_MIN=$(echo "$X_MIN_GLOBAL + $i * ($TILE_WIDTH - $OVERLAP)" | bc)
        Y_MIN=$(echo "$Y_MIN_GLOBAL + $j * ($TILE_HEIGHT - $OVERLAP)" | bc)
        X_MAX=$(echo "$X_MIN + $TILE_WIDTH" | bc)
        Y_MAX=$(echo "$Y_MIN + $TILE_HEIGHT" | bc)

        # Ensure we don't exceed global bounds
        X_MAX=$(echo "if($X_MAX > $X_MAX_GLOBAL) $X_MAX_GLOBAL else $X_MAX" | bc)
        Y_MAX=$(echo "if($Y_MAX > $Y_MAX_GLOBAL) $Y_MAX_GLOBAL else $Y_MAX" | bc)

        # Create a job submission script for this tile
        JOB_SCRIPT="jobs/job_${i}_${j}.sh"
        echo "#!/bin/bash" > $JOB_SCRIPT
        echo "source activate segger_dev" >> $JOB_SCRIPT  # Activate your conda environment if needed
        echo "python -m segger.cli.create_dataset \\" >> $JOB_SCRIPT
        echo "  --dataset_type xenium \\" >> $JOB_SCRIPT
        echo "  --sample_tag $SAMPLE_TAG \\" >> $JOB_SCRIPT
        echo "  --dataset_dir $DATASET_DIR \\" >> $JOB_SCRIPT
        echo "  --data_dir $DATA_DIR \\" >> $JOB_SCRIPT
        echo "  --transcripts_file $TRANSCRIPTS_FILE \\" >> $JOB_SCRIPT
        echo "  --boundaries_file $BOUNDARIES_FILE \\" >> $JOB_SCRIPT
        echo "  --method kd_tree \\" >> $JOB_SCRIPT
        echo "  --x_min $X_MIN --y_min $Y_MIN --x_max $X_MAX --y_max $Y_MAX" >> $JOB_SCRIPT
        # chmod +x $JOB_SCRIPT

        # Submit the job to the cluster
        # bsub -R "rusage[mem=200G]" -q long -o "$LOG_DIR/job_${i}_${j}.log" < $JOB_SCRIPT
    done
done

# Clean up
rm bounds.txt
