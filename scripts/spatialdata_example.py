from segger.data.io import XeniumSample, SpatialDataSample
from pathlib import Path

# Paths to Xenium sample data and where to store Segger data
segger_data_dir = Path("data/segger_data/")
# raw xenium data
# xenium_data_dir = Path('data/xenium_2.0.0_io/data')
# xenium_data_dir = Path('data/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny_outs')

# spatialdata zarr data
xenium_data_dir = Path("data/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny_outs.zarr")

# Setup Xenium sample to create dataset
# xs = XeniumSample(verbose=False)
# xs.set_file_paths(
#     # raw xenium data
#     transcripts_path=xenium_data_dir / 'transcripts.parquet',
#     boundaries_path=xenium_data_dir / 'nucleus_boundaries.parquet',
# )
xs = SpatialDataSample(verbose=False, feature_name="feature_name")
xs.set_file_paths(
    # spatialdata zarr data
    transcripts_path=xenium_data_dir / "points/transcripts/points.parquet",
    boundaries_path=xenium_data_dir / "shapes/nucleus_boundaries/shapes.parquet",
)
xs.set_metadata()

import shutil

if segger_data_dir.exists() and "segger_data" in str(segger_data_dir):
    shutil.rmtree(str(segger_data_dir))

try:
    xs.save_dataset_for_segger(
        processed_dir=segger_data_dir,
        r_tx=5,
        k_tx=15,
        x_size=120,
        y_size=120,
        d_x=100,
        d_y=100,
        margin_x=10,
        margin_y=10,
        scale_boundaries=1,
        num_workers=1,  # change to you number of CPUs
        # val_prob=0.5,
    )
except AssertionError as err:
    print(f"Dataset already exists at {segger_data_dir}")
