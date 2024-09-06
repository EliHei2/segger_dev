import scanpy as sc
from segger.data.io import *
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from segger.data import XeniumSample

# Paths for raw and processed data
raw_data_dir = Path('data_raw/xenium/')
processed_data_dir = Path('data_tidy/pyg_datasets')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

# Ensure directories exist
raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Define paths for transcripts and nuclei data
transcripts_path = raw_data_dir / "transcripts.parquet"
nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"
scRNAseq_path = '/omics/groups/OE0606/internal/tangy/tasks/schier/data/atlas_filtered.h5ad'

# Step 1: Load scRNA-seq data using Scanpy and subsample for efficiency
scRNAseq = sc.read(scRNAseq_path)
sc.pp.subsample(scRNAseq, 0.1)

# Step 2: Calculate gene cell type abundance embedding from scRNA-seq data
celltype_column = 'celltype_minor'
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, celltype_column)

# Step 3: Create a XeniumSample instance for spatial transcriptomics processing
xenium_sample = XeniumSample()

# Step 4: Load transcripts and include the calculated cell type abundance embedding
xenium_sample.load_transcripts(
    base_path=raw_data_dir,
    sample=sample_tag,
    transcripts_filename='transcripts.parquet',
    file_format="parquet",
    additional_embeddings={"cell_type_abundance": gene_celltype_abundance_embedding}
)

# Step 5: Set the embedding to "cell_type_abundance" to use it in further processing
xenium_sample.set_embedding("cell_type_abundance")

# Step 6: Load nuclei data to define boundaries
xenium_sample.load_boundaries(path=nuclei_path, file_format='parquet')

# Optional Step: Extract a specific region (bounding box) of the sample (uncomment if needed)
# xenium_sample.get_bounding_box(x_min=1000, y_min=1000, x_max=1360, y_max=1360, in_place=True)

# Step 7: Build PyTorch Geometric (PyG) data from a tile of the dataset
start_time = time.time()
pyg_data = xenium_sample.build_pyg_data_from_tile(
    boundaries_df=xenium_sample.boundaries_df,
    transcripts_df=xenium_sample.transcripts_df,
    r_tx=20,
    k_tx=20,
    use_precomputed=False,  # Set to True if precomputed graph available
    workers=1
)
end_time = time.time()
print(f"Time to build PyG data: {end_time - start_time} seconds")

# Step 8: Save the dataset in a format compatible with Segger using tiling
start_time = time.time()
xenium_sample.save_dataset_for_segger(
    processed_dir=processed_data_dir / 'embedding',
    x_size=360,
    y_size=360,
    d_x=180,
    d_y=180,
    margin_x=10,
    margin_y=10,
    compute_labels=False,  # Set to True if you need to compute labels
    r_tx=5,
    k_tx=5,
    val_prob=0.1,
    test_prob=0.2,
    neg_sampling_ratio_approx=5,
    sampling_rate=1,
    num_workers=1,
    receptive_field={
        "k_bd": 4,
        "dist_bd": 15,
        "k_tx": 5,
        "dist_tx": 5,
    },
    use_precomputed=False  # Option to use precomputed edges (if available)
)
end_time = time.time()
print(f"Time to save dataset: {end_time - start_time} seconds")
