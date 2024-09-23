import scanpy as sc
from segger.data.io import *
from segger.data.utils import *
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from segger.data import XeniumSample, SpatialTranscriptomicsSample
from segger.data.utils import comp
from dask import delayed
import geopandas as gpd

# Paths for raw and processed data
raw_data_dir = Path('data_raw/xenium/')
processed_data_dir = Path('data_tidy/pyg_datasets')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

# Ensure directories exist
raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Define paths for transcripts and nuclei data
transcripts_path = raw_data_dir / sample_tag / "transcripts.parquet"
nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"

# Step 1: Set paths for transcripts and boundaries
xenium_sample = XeniumSample()
xenium_sample.set_file_paths(transcripts_path=transcripts_path, boundaries_path=nuclei_path)
xenium_sample.set_metadata()



# Step 2: Use save_dataset_for_segger directly to handle tiling, processing, and lazy reading of each tile
start_time = time.time()
xenium_sample.save_dataset_for_segger(
    processed_dir=processed_data_dir / 'fixed_0911',
    x_size=300,
    y_size=300,
    d_x=280,
    d_y=280,
    margin_x=10,
    margin_y=10,
    compute_labels=True,  # Set to True if you need to compute labels
    r_tx=5,
    k_tx=10,
    val_prob=0.4,
    test_prob=0.1,
    neg_sampling_ratio_approx=5,
    sampling_rate=1,
    num_workers=1,
)
end_time = time.time()
print(f"Time to save dataset: {end_time - start_time} seconds")