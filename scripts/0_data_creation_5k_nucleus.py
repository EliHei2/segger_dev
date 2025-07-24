from segger.data.parquet.sample import STSampleParquet
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import math
import numpy as np
from segger.data.parquet._utils import get_polygons_from_xy

"""
This script preprocesses Xenium spatial transcriptomics data for SEGGER cell segmentation model.


Parameters are set properly for a 5K panel.

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



XENIUM_DATA_DIR = Path(
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/xenium_seg_kit/human_CRC_real"
)
SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/human_CRC_seg_nuclei")
SCRNASEQ_FILE = Path(
    "data_tidy/Human_CRC/scRNAseq.h5ad"
)
CELLTYPE_COLUMN = "Level1" # change this to your column name
scrnaseq = sc.read(SCRNASEQ_FILE)



# subsample the scRNAseq if needed
sc.pp.subsample(scrnaseq, 0.1)
scrnaseq.var_names_make_unique()


# Calculate gene-celltype embeddings from reference data
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
    scrnaseq,
    CELLTYPE_COLUMN
)

# Initialize spatial transcriptomics sample object
sample = STSampleParquet(
    base_dir=XENIUM_DATA_DIR,
    n_workers=4,
    sample_type="xenium",
    weights=gene_celltype_abundance_embedding
)


sample.save(
    data_dir=SEGGER_DATA_DIR,
    k_bd=3,  # Number of boundary points to connect
    dist_bd=10,  # Maximum distance for boundary connections
    k_tx=5,  # Use calculated optimal transcript neighbors
    dist_tx=5,  # Use calculated optimal search radius
    tile_size=10000,  # Tile size for processing
    # tile_height=50,
    neg_sampling_ratio=10.,  # 5:1 negative:positive samples
    frac=1.0,  # Use all data
    val_prob=0.3,  # 30% validation set
    test_prob=0,  # No test set
)
