import scanpy as sc
from segger.data.io import *
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

# Load your data and initial setup
raw_data_dir = Path('data_raw/xenium/')
processed_data_dir = Path('data_tidy/pyg_datasets')
figures_dir = Path('figures/')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

# Create the figures directory if it doesn't exist
figures_dir.mkdir(parents=True, exist_ok=True)

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

# Set initial bounds
xenium_sample._set_bounds()

# Define the methods to compare
methods = ['kd_tree', 'hnsw', 'faiss_cpu', 'faiss_gpu']

# Initialize an empty dictionary to store timings for each method and subset size
all_timings = []

# Step 1: Measure the time taken by each method on progressively smaller datasets
x_range = (xenium_sample.x_max - xenium_sample.x_min)
y_range = (xenium_sample.y_max - xenium_sample.y_min)

while x_range > 10 and y_range > 10:  # Arbitrary cutoff to avoid excessively small subsets
    # Update bounds for the current subset
    xenium_sample.get_bounding_box(
        x_min=xenium_sample.x_min,
        y_min=xenium_sample.y_min,
        x_max=xenium_sample.x_min + x_range,
        y_max=xenium_sample.y_min + y_range,
        in_place=True
    )
    
    # Record the number of transcripts
    num_transcripts = len(xenium_sample.transcripts_df)
    
    # Measure the time for each method
    timings = {}
    for method in methods:
        base_method = method
        if 'faiss' in method:
            gpu = 'gpu' in method  # Determine if GPU should be used for FAISS
            base_method = method.split('_')[0] 
        else:
            gpu = False  # RAPIDS and cuGraph always use GPU, no need for the flag
        
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
        print(f"{method} method took {elapsed_time:.4f} seconds on {num_transcripts} transcripts")
    
    # Store the results
    timings['num_transcripts'] = num_transcripts
    all_timings.append(timings)
    
    # Reduce the bounding box size by half
    x_range /= 2
    y_range /= 2

# Convert the results to a DataFrame
timings_df = pd.DataFrame(all_timings)

# Save the results to a CSV file
timings_df.to_csv('timings_results_by_subset.csv', index=False)

# Step 2: Plot the results with color-blind-friendly colors
color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']  # A color-blind-friendly palette

plt.figure(figsize=(10, 6))
for method, color in zip(methods, color_palette):
    plt.plot(timings_df['num_transcripts'], timings_df[method], label=method, color=color)

plt.xlabel('Number of Transcripts')
plt.ylabel('Time (seconds)')
plt.title('Method Timing vs. Number of Transcripts')
plt.legend()
plt.grid(True)

# Save the plot as a PDF
plt.savefig(figures_dir / 'method_timing_vs_transcripts.pdf', format='pdf')

plt.show()
