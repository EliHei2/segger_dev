from segger.data.parquet.sample import STSampleParquet
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import math
import numpy as np
from segger.data.parquet._utils import get_polygons_from_xy

xenium_data_dir = Path('data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs/')
segger_data_dir = Path('data_tidy/pyg_datasets/bc_rep1_emb_200_final')


scrnaseq_file = Path('/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad')
celltype_column = 'celltype_minor'
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
    sc.read(scrnaseq_file),
    celltype_column
)

sample = STSampleParquet(
    base_dir=xenium_data_dir,
    n_workers=4,
    sample_type="xenium",
    weights=gene_celltype_abundance_embedding,  # uncomment if gene-celltype embeddings are available
)

transcripts = pd.read_parquet(xenium_data_dir / "transcripts.parquet", filters=[[("overlaps_nucleus", "=", 1)]])
boundaries = pd.read_parquet(xenium_data_dir / "nucleus_boundaries.parquet")

sizes = transcripts.groupby("cell_id").size()
polygons = get_polygons_from_xy(boundaries, "vertex_x", "vertex_y", "cell_id")
densities = polygons[sizes.index].area / sizes
bd_width = polygons.minimum_bounding_radius().median() * 2

# 1/4 median boundary diameter
dist_tx = bd_width / 4
# 90th percentile density of bounding circle with radius=dist_tx
k_tx = math.ceil(np.quantile(dist_tx**2 * np.pi * densities, 0.9))

print(k_tx)
print(dist_tx)


sample.save(
      data_dir=segger_data_dir,
      k_bd=3,
      dist_bd=15,
      k_tx=3,
      dist_tx=5,
      tile_width=200,
      tile_height=200,
      neg_sampling_ratio=5.0,
      frac=1.0,
      val_prob=0.3,
      test_prob=0,
)


xenium_data_dir = Path('data_tidy/bc_5k')
segger_data_dir = Path('data_tidy/pyg_datasets/bc_5k_emb_new')



sample = STSampleParquet(
    base_dir=xenium_data_dir,
    n_workers=8,
    sample_type='xenium',
    weights=gene_celltype_abundance_embedding, # uncomment if gene-celltype embeddings are available
)


transcripts = pd.read_parquet(xenium_data_dir / "transcripts.parquet", filters=[[("overlaps_nucleus", "=", 1)]])
boundaries = pd.read_parquet(xenium_data_dir / "nucleus_boundaries.parquet")

sizes = transcripts.groupby("cell_id").size()
polygons = get_polygons_from_xy(boundaries, "vertex_x", "vertex_y", "cell_id")
densities = polygons[sizes.index].area / sizes
bd_width = polygons.minimum_bounding_radius().median() * 2

# 1/4 median boundary diameter
dist_tx = bd_width / 4
# 90th percentile density of bounding circle with radius=dist_tx
k_tx = math.ceil(np.quantile(dist_tx**2 * np.pi * densities, 0.9))

print(k_tx)
print(dist_tx)


sample.save(
      data_dir=segger_data_dir,
      k_bd=3,
      dist_bd=15.0,
      k_tx=15,
      dist_tx=3,
      tile_size=50_000, 
      neg_sampling_ratio=5.0,
      frac=0.1,
      val_prob=0.1,
      test_prob=0.1,
)


