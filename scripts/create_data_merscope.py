from segger.data.parquet.sample import STSampleParquet, STInMemoryDataset
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import math
import numpy as np
from segger.data.parquet._utils import get_polygons_from_xy

"""
This script preprocesses MERSCOPE spatial transcriptomics data for SEGGER cell segmentation model.

Key steps:
1. Data Loading:
   - Loads scRNA-seq reference data to create gene-celltype embeddings
   - Imports MERSCOPE transcripts and nucleus boundaries
   
2. Parameter Optimization:
   - Calculates optimal neighborhood parameters based on tissue characteristics
   - dist_tx: Sets transcript neighbor search radius to 1/4 of typical nucleus size
   - k_tx: Determines number of transcripts to sample based on local density
   
3. Dataset Creation:
   - Filters transcripts to those overlapping nuclei
   - Creates graph connections between nearby transcripts
   - Splits data into training/validation sets
   - Saves in PyG format for SEGGER training

Usage:
- Input: Raw MERSCOPE data (transcripts.parquet, nucleus_boundaries.parquet)
- Output: Processed dataset with graph structure and embeddings
"""

# Define data paths
# MERSCOPE_DATA_DIR = Path('/omics/odcf/analysis/OE0606_projects_temp/MERSCOPE_projects/20241209_MERSCOPE5k_CNSL_BrM/20241209_MERSCOPE5k_CNSL_BrM/output-XETG00078__0041719__Region_1__20241203__142052')
# SEGGER_DATA_DIR = Path('data_tidy/pyg_datasets/CNSL_5k')
# # SCRNASEQ_FILE = Path('/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad')
# CELLTYPE_COLUMN = 'celltype_minor'


MERSCOPE_DATA_DIR = Path('data_raw/merscope/processed/')
SEGGER_DATA_DIR = Path('data_tidy/pyg_datasets/merscope_liver')
# SCRNASEQ_FILE = Path('/omics/groups/OE0606/internal/mimmo/MERSCOPE/notebooks/data/scData/bh/bh_mng_scdata_20250306.h5ad')
# CELLTYPE_COLUMN = 'annot_v1'

# Calculate gene-celltype embeddings from reference data
# gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
#     sc.read(SCRNASEQ_FILE),
#     CELLTYPE_COLUMN
# )

# Initialize spatial transcriptomics sample object
sample = STSampleParquet(
    base_dir=MERSCOPE_DATA_DIR,
    n_workers=4,
    sample_type="merscope",
    buffer_ratio=1,
    # weights=gene_celltype_abundance_embedding
)

# Load and filter data


# Save processed dataset for SEGGER
# Parameters:
# - k_bd/dist_bd: Control nucleus boundary point connections
# - k_tx/dist_tx: Control transcript neighborhood connections
# - tile_width/height: Size of spatial tiles for processing
# - neg_sampling_ratio: Ratio of negative to positive samples
# - val_prob: Fraction of data for validation
sample.save_debug(
    data_dir=SEGGER_DATA_DIR,
    k_bd=3,  # Number of boundary points to connect
    dist_bd=15,  # Maximum distance for boundary connections
    k_tx=5,  # Use calculated optimal transcript neighbors
    dist_tx=20,  # Use calculated optimal search radius
    tile_width=500,  # Tile size for processing
    tile_height=500,
    neg_sampling_ratio=5.0,  # 5:1 negative:positive samples
    frac=1.0,  # Use all data
    val_prob=0.3,  # 30% validation set
    test_prob=0,  # No test set
)