import scanpy as sc
from segger.data.utils import *
# Assuming we have an AnnData object `adata` from scRNA-seq analysis


raw_data_dir = Path('data_raw/xenium')
processed_data_dir = Path('data_tidy/pyg_datasets')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

transcripts_path = raw_data_dir / "transcripts.parquet"
nuclei_path = raw_data_dir / "nucleus_boundaries.csv.gz"
scRNAseq_path = Path('data_tidy/benchmarks/xe_rep1_bc') / 'scRNAseq.h5ad'

scRNAseq = sc.read(scRNAseq_path)


# Step 1: Calculate the gene cell type abundance embedding
celltype_column = 'celltype_major'
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, celltype_column)

# Step 2: Create a XeniumSample instance
xenium_sample = XeniumSample()

# Step 3: Load transcripts and include the cell type abundance embedding
xenium_sample.load_transcripts(
    base_path=Path(raw_data_dir),
    sample=sample_tag,
    transcripts_filename='transcripts.parquet',
    file_format="parquet",
    additional_embeddings={"cell_type_abundance": gene_celltype_abundance_embedding}
)

# Step 4: Set the embedding to "cell_type_abundance"
xenium_sample.set_embedding("cell_type_abundance")

# Step 5: Build PyG data with the selected embedding
nuclei_df = pd.DataFrame()  # Replace with actual nuclei data
pyg_data = xenium_sample.build_pyg_data_from_tile(nuclei_df, xenium_sample.transcripts_df)

# Now, `pyg_data` will use the cell type abundance embedding for the transcripts

