import scanpy as sc
from segger.data.utils import *
# Assuming we have an AnnData object `adata` from scRNA-seq analysis


raw_data_dir = Path('data_raw/xenium/')
processed_data_dir = Path('data_tidy/pyg_datasets')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

transcripts_path = raw_data_dir / "transcripts.parquet"
nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"
scRNAseq_path = '/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad'

scRNAseq = sc.read(scRNAseq_path)


# Step 1: Calculate the gene cell type abundance embedding
celltype_column = 'celltype_minor'
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

xenium_sample.load_nuclei(path=nuclei_path, file_format='parquet')

xenium_sample.get_bounding_box(x_max=1000, y_max=1000, in_place=True)

xenium_sample.save_dataset_for_segger(
        processed_dir=processed_data_dir / (sample_tag +'_emb'),
        x_size=200,
        y_size=200,
        d_x=180,
        d_y=180,
        margin_x=20,
        margin_y=20,
        r_tx=10,
        val_prob=.2,
        test_prob=.2,
        compute_labels=True,
        sampling_rate=1,
        num_workers=0,
        receptive_field={
            "k_nc": 5,
            "dist_nc": 10,
            "k_tx": 10,
            "dist_tx": 3,
        },
    )

# Now, `pyg_data` will use the cell type abundance embedding for the transcripts

