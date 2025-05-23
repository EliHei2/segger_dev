from segger.data.parquet.sample import STSampleParquet, STInMemoryDataset
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import math
import numpy as np
from segger.data.parquet._utils import get_polygons_from_xy

"""
This script preprocesses Xenium spatial transcriptomics data for SEGGER cell segmentation model.

Key steps:
1. Data Loading:
   - Loads scRNA-seq reference data to create gene-celltype embeddings
   - Imports Xenium transcripts and nucleus boundaries
   
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
- Input: Raw Xenium data (transcripts.parquet, nucleus_boundaries.parquet)
- Output: Processed dataset with graph structure and embeddings
"""

# Define data paths
# XENIUM_DATA_DIR = Path('/omics/odcf/analysis/OE0606_projects_temp/xenium_projects/20241209_Xenium5k_CNSL_BrM/20241209_Xenium5k_CNSL_BrM/output-XETG00078__0041719__Region_1__20241203__142052')
# SEGGER_DATA_DIR = Path('data_tidy/pyg_datasets/CNSL_5k')
# # SCRNASEQ_FILE = Path('/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad')
# CELLTYPE_COLUMN = 'celltype_minor'


XENIUM_DATA_DIR = Path("data_raw/cosmx/human_pancreas/processed/")
SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/cosmx_pancreas_degbugged_")
# SCRNASEQ_FILE = Path('/omics/groups/OE0606/internal/mimmo/Xenium/notebooks/data/scData/bh/bh_mng_scdata_20250306.h5ad')
# CELLTYPE_COLUMN = 'annot_v1'

# Calculate gene-celltype embeddings from reference data
# gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
#     sc.read(SCRNASEQ_FILE),
#     CELLTYPE_COLUMN
# )

# Initialize spatial transcriptomics sample object
sample = STSampleParquet(
    base_dir=XENIUM_DATA_DIR,
    n_workers=4,
    sample_type="cosmx",
    scale_factor=1.1,
    # weights=gene_celltype_abundance_embedding
)

# Load and filter data
transcripts = pd.read_parquet(XENIUM_DATA_DIR / "transcripts.parquet")
boundaries = pd.read_parquet(XENIUM_DATA_DIR / "nucleus_boundaries.parquet")

# Calculate optimal neighborhood parameters
transcript_counts = transcripts.groupby("cell").size()
nucleus_polygons = get_polygons_from_xy(
    boundaries, "x_global_px", "y_global_px", "cell"
)


cells = list(set(transcript_counts.index) &  set(nucleus_polygons.index))
nucleus_polygons = nucleus_polygons[cells]
transcript_counts = transcript_counts[cells]

transcript_densities = (
    nucleus_polygons[transcript_counts.index].area / transcript_counts
)
nucleus_diameter = nucleus_polygons.minimum_bounding_radius().median() * 2

# Set neighborhood parameters
dist_tx = nucleus_diameter / 4  # Search radius = 1/4 nucleus diameter
k_tx = math.ceil(
    np.quantile(dist_tx**2 * np.pi * transcript_densities, 0.9)
)  # Sample size based on 90th percentile density

print(f"Calculated parameters: k_tx={k_tx}, dist_tx={dist_tx:.2f}")

# Save processed dataset for SEGGER
# Parameters:
# - k_bd/dist_bd: Control nucleus boundary point connections
# - k_tx/dist_tx: Control transcript neighborhood connections
# - tile_width/height: Size of spatial tiles for processing
# - neg_sampling_ratio: Ratio of negative to positive samples
# - val_prob: Fraction of data for validation
sample.save(
    data_dir=SEGGER_DATA_DIR,
    k_bd=3,  # Number of boundary points to connect
    dist_bd=15,  # Maximum distance for boundary connections
    k_tx=10,  # Use calculated optimal transcript neighbors
    dist_tx=10,  # Use calculated optimal search radius
    tile_width=200,  # Tile size for processing,
    tile_height=200,  # Tile size for processing
    neg_sampling_ratio=10.0,  # 5:1 negative:positive samples
    frac=1.0,  # Use all data
    val_prob=0.3,  # 30% validation set
    test_prob=0,  # No test set
)
