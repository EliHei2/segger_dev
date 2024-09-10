import scanpy as sc
from segger.data.io import *
from segger.data.utils import *
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from segger.data import XeniumSample, SpatialTranscriptomicsSample
from dask import delayed
import geopandas as gpd

# Paths for raw and processed data
raw_data_dir = Path('data_raw/xenium/')
processed_data_dir = Path('data_tidy/pyg_datasets')
sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

# Ensure directories exist
raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Define paths for transcripts and nuclei data
transcripts_path = raw_data_dir / sample_tag / "transcripts.parquet"
nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"

# Step 1: Set paths for transcripts and boundaries
xenium_sample = XeniumSample()
xenium_sample.set_file_paths(transcripts_path=transcripts_path, boundaries_path=nuclei_path)
xenium_sample.set_metadata()



# Step 2: Use save_dataset_for_segger directly to handle tiling, processing, and lazy reading of each tile
start_time = time.time()
xenium_sample.save_dataset_for_segger(
    processed_dir=processed_data_dir / 'clean_parallel',
    x_size=240,
    y_size=240,
    d_x=180,
    d_y=180,
    margin_x=10,
    margin_y=10,
    compute_labels=True,  # Set to True if you need to compute labels
    r_tx=5,
    k_tx=5,
    val_prob=0.1,
    test_prob=0.2,
    neg_sampling_ratio_approx=5,
    sampling_rate=1,
    num_workers=2,
    # receptive_field={
    #     "k_bd": 4,
    #     "dist_bd": 15,
    #     "k_tx": 5,
    #     "dist_tx": 5,
    # },
    # use_precomputed=False  # Option to use precomputed edges (if available)
)
end_time = time.time()
print(f"Time to save dataset: {end_time - start_time} seconds")




# import scanpy as sc
# from segger.data.io import *
# from segger.data.utils import *
# from pathlib import Path
# import time
# import pandas as pd
# import matplotlib.pyplot as plt
# from segger.data import XeniumSample, SpatialTranscriptomicsSample
# from dask import delayed

# # Paths for raw and processed data
# raw_data_dir = Path('data_raw/xenium/')
# processed_data_dir = Path('data_tidy/pyg_datasets')
# sample_tag = "Xenium_FFPE_Human_Breast_Cancer_Rep1"

# # Ensure directories exist
# raw_data_dir.mkdir(parents=True, exist_ok=True)
# processed_data_dir.mkdir(parents=True, exist_ok=True)

# # Define paths for transcripts and nuclei data
# transcripts_path = raw_data_dir / "transcripts.parquet"
# nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"
# # scRNAseq_path = '/omics/groups/OE0606/internal/tangy/tasks/schier/data/atlas_filtered.h5ad'

# # # # Step 1: Load scRNA-seq data using Scanpy and subsample for efficiency
# # scRNAseq = sc.read(scRNAseq_path)
# # sc.pp.subsample(scRNAseq, 0.1)

# # # Step 2: Calculate gene cell type abundance embedding from scRNA-seq data
# # celltype_column = 'celltype_minor'
# # gene_celltype_abundance_embedding = delayed(calculate_gene_celltype_abundance_embedding)(scRNAseq, celltype_column)

# # Step 3: Create a XeniumSample instance for spatial transcriptomics processing
# xenium_sample = XeniumSample()

# # Step 4: Load transcripts and include the calculated cell type abundance embedding
# xenium_sample.load_transcripts(
#     base_path=raw_data_dir,
#     sample=sample_tag,
#     transcripts_filename='transcripts.parquet',
#     file_format="parquet",
#     # additional_embeddings={"cell_type_abundance": gene_celltype_abundance_embedding}
# )

# # Step 5: Set the embedding to "cell_type_abundance" to use it in further processing
# xenium_sample.set_embedding("one_hot")

# # Step 6: Load nuclei data to define boundaries
# xenium_sample.load_boundaries(path=nuclei_path, file_format='parquet')

# # Optional Step: Extract a specific region (bounding box) of the sample (uncomment if needed)
# # xenium_sample.get_bounding_box(x_min=1000, y_min=1000, x_max=1360, y_max=1360, in_place=True)

# boundaries_df=xenium_sample.boundaries_df
# keys = {
#     'vertex_x': 'vertex_x',
#     'vertex_y': 'vertex_y',
#     'cell_id': 'cell_id'
# }


# delayed_tasks = []
# for  cell_id, group in boundaries_df.compute().groupby('cell_id'):
#     delayed_task = delayed(SpatialTranscriptomicsSample.create_scaled_polygon)(
#         group,  # Compute each group
#         scale_factor=1,   # Pass scale factor
#         keys=keys         # Pass keys
#     )
#     delayed_tasks.append(delayed_task)

# # Compute the results
# polygons_list = delayed(delayed_tasks).compute()

# # Convert the results back to a single GeoDataFrame
# polygons_gdf = gpd.GeoDataFrame(pd.concat(polygons_list, ignore_index=True))


# # def process_group(group, scale_factor, keys):
# #     return delayed(create_scaled_polygon)(group, scale_factor, keys)


# # # Define the keys dictionary
# # keys = {
# #     'cell_id': cell_id_column,
# #     'vertex_x': vertex_x_column,
# #     'vertex_y': vertex_y_column
# # }
# # meta={'geometry': 'object', 'cell_id': 'str'}

# # # Apply the function on the group
# # polygons_df = boundaries_df.groupby(cell_id_column).apply(
# #     process_group,  # Use the defined function instead of lambda
# #     scale_factor=1, 
# #     keys=keys,
# #     meta=meta
# #       # Explicit meta definition
# # )



# # Step 8: Save the dataset in a format compatible with Segger using tiling
# start_time = time.time()
# xenium_sample.save_dataset_for_segger(
#     processed_dir=processed_data_dir / 'embedding',
#     x_size=360,
#     y_size=360,
#     d_x=180,
#     d_y=180,
#     margin_x=10,
#     margin_y=10,
#     compute_labels=False,  # Set to True if you need to compute labels
#     r_tx=5,
#     k_tx=5,
#     val_prob=0.1,
#     test_prob=0.2,
#     neg_sampling_ratio_approx=5,
#     sampling_rate=1,
#     num_workers=1,
#     receptive_field={
#         "k_bd": 4,
#         "dist_bd": 15,
#         "k_tx": 5,
#         "dist_tx": 5,
#     },
#     use_precomputed=False  # Option to use precomputed edges (if available)
# )
# end_time = time.time()
# print(f"Time to save dataset: {end_time - start_time} seconds")
