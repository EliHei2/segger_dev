import scanpy as sc
from segger.data.io import *
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

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

sc.pp.subsample(scRNAseq, 0.1)

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

# Load nuclei data
xenium_sample.load_boundaries(path=nuclei_path, file_format='parquet')

# xenium_sample.get_bounding_box(x_min = 1000, y_min = 1000, x_max=2000, y_max=2000, in_place=True)


xenium_sample.precompute_tx_tx_graph(k=5, dist=10, workers = 10)


data = xenium_sample.build_pyg_data_from_tile(
            boundaries_df=xenium_sample.boundaries_df,
            transcripts_df=xenium_sample.transcripts_df,
            r_tx = 20,
            k_tx = 20,
            use_precomputed=True,
            workers=1
        )


# Crop to a smaller bounding box to speed up the comparison


# Compare the speed of different methods
methods = ['kd_tree', 'hnsw', 'faiss_cpu', 'faiss_gpu']

# methods = ['faiss_cpu', 'kd_tree']
timings = {}

# Measure the time taken by each method
for method in methods:
    base_method = method
    if 'faiss' in method:
        gpu = 'gpu' in method  # Determine if GPU should be used for FAISS
        base_method = method.split('_')[0] 
    else:
        gpu = False  # RAPIDS and cuGraph always use GPU, no need for the flag
    
     # Extract the base method (e.g., 'faiss', 'rapids', etc.)
    
    start_time = time.time()
    data = xenium_sample.build_pyg_data_from_tile(
        boundaries_df=xenium_sample.boundaries_df,
        transcripts_df=xenium_sample.transcripts_df,
        compute_labels=True,
        method=base_method,
        gpu=gpu,
        workers=1
    )
    elapsed_time = time.time() - start_time
    timings[method] = elapsed_time
    print(f"{method} method took {elapsed_time:.4f} seconds")

# Save timings to a CSV file
timings_df = pd.DataFrame(list(timings.items()), columns=['Method', 'Time'])
timings_df.to_csv('timings_results.csv', index=False)

# Generate a bar plot of the timings
plt.figure(figsize=(10, 6))
plt.bar(timings_df['Method'], timings_df['Time'], color='skyblue')
plt.xlabel('Method')
plt.ylabel('Time (seconds)')
plt.title('Timing Comparison of Different Methods')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('timings_comparison_plot.png')

print("Results saved to 'timings_results.csv' and 'timings_comparison_plot.png'.")
