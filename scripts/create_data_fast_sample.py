from segger.data.parquet.sample import STSampleParquet
from path import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc

xenium_data_dir = Path('data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs/')
segger_data_dir = Path('data_tidy/pyg_datasets/bc_fast_data_emb_major')


scrnaseq_file = Path('data_tidy/benchmarks/xe_rep1_bc/scRNAseq.h5ad')
celltype_column = 'celltype_major'
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
    sc.read(scrnaseq_file),
    celltype_column
)

sample = STSampleParquet(
    base_dir=xenium_data_dir,
    n_workers=4,
    sample_type='xenium',
    weights=gene_celltype_abundance_embedding, # uncomment if gene-celltype embeddings are available
)

sample.save(
      data_dir=segger_data_dir,
      k_bd=3,
      dist_bd=15.0,
      k_tx=20,
      dist_tx=3,
      tile_width=220,
      tile_height=220,
      neg_sampling_ratio=5.0,
      frac=1.0,
      val_prob=0.1,
      test_prob=0.1,
)