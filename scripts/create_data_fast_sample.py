from segger.data.parquet.sample import STSampleParquet
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import math
import numpy as np
from segger.data.parquet._utils import get_polygons_from_xy
import anndata as ad
from typing import Dict, List, Tuple
from itertools import combinations
from segger.data.parquet._utils import find_markers, find_mutually_exclusive_genes
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


XENIUM_DATA_DIR = Path(
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs"
)
SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/xe_bc_rep1_loss_emb2")
SCRNASEQ_FILE = Path(
    "/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad"
)
CELLTYPE_COLUMN = "celltype_minor"
scrnaseq = sc.read(SCRNASEQ_FILE)
sc.pp.subsample(scrnaseq, 0.25)
scrnaseq.var_names_make_unique()
sc.pp.log1p(scrnaseq)
sc.pp.normalize_total(scrnaseq)
# Calculate gene-celltype embeddings from reference data
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
    scrnaseq,
    CELLTYPE_COLUMN
)



# markers = find_markers(scrnaseq, cell_type_column="celltype_minor", pos_percentile=20, neg_percentile=20, percentage=50)


    

# Initialize spatial transcriptomics sample object
sample = STSampleParquet(
    base_dir=XENIUM_DATA_DIR,
    n_workers=10,
    sample_type="xenium",
    # scale_factor=0.8,
    weights=gene_celltype_abundance_embedding
)




genes = list(set(scrnaseq.var_names) & set(sample.transcripts_metadata['feature_names']))
markers = find_markers(scrnaseq[:,genes], cell_type_column="celltype_minor", pos_percentile=90, neg_percentile=20, percentage=20)
# Find mutually exclusive genes based on scRNAseq data
exclusive_gene_pairs = find_mutually_exclusive_genes(
    adata=scrnaseq,
    markers=markers,
    cell_type_column="celltype_minor"
)


sample.save(
    data_dir=SEGGER_DATA_DIR,
    k_bd=3,  # Number of boundary points to connect
    dist_bd=15,  # Maximum distance for boundary connections
    k_tx=20,  # Use calculated optimal transcript neighbors
    dist_tx=5,  # Use calculated optimal search radius
    k_tx_ex=20,  # Use calculated optimal transcript neighbors
    dist_tx_ex=20,  # Use calculated optimal search radius
    tile_size=10_000,  # Tile size for processing
    # tile_height=100,
    neg_sampling_ratio=10.0,  # 5:1 negative:positive samples
    frac=1.0,  # Use all data
    val_prob=0.3,  # 30% validation set
    test_prob=0,  # No test set
    # k_tx_ex=100,  # Use calculated optimal transcript neighbors
    # dist_tx_ex=20,  # Use calculated optimal search radius
    mutually_exclusive_genes=exclusive_gene_pairs
)
