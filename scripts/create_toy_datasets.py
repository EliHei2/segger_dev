import dask.dataframe as dd
import tifffile as tiff  # Use tifffile instead of PIL for OME-TIFF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Define the paths to the input Parquet and TIFF files (update these paths to match your file locations)
transcripts_file = "data_raw/xenium/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs/transcripts.parquet"
nuclei_file = "data_raw/xenium/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs/nucleus_boundaries.parquet"
cell_boundaries_file = "data_raw/xenium/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs/cell_boundaries.parquet"
morphology_tiff_file = "data_raw/xenium/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs/morphology.ome.tif"

# Define the output directory for the toy dataset
output_dir = Path("data_raw/package_toy_data/xenium_pancreas_cancer")
output_dir.mkdir(parents=True, exist_ok=True)

def find_fovs_in_square(transcripts_file, square_size=3):
    print(f"Loading transcripts from {transcripts_file} using Dask...")
    transcripts_df = dd.read_parquet(transcripts_file)
    fov_list = transcripts_df['fov_name'].drop_duplicates().compute().tolist()
    sorted_fovs = sorted(fov_list)
    middle_index = len(sorted_fovs) // 2
    half_size = square_size // 2
    start_index = max(middle_index - half_size, 0)
    end_index = min(middle_index + half_size + 1, len(sorted_fovs))
    selected_fovs = sorted_fovs[start_index:end_index]
    if len(selected_fovs) < square_size ** 2:
        print("Warning: The selected square is smaller than expected due to slide boundaries.")
    print(f"Selected FOVs: {selected_fovs}")
    return selected_fovs, transcripts_df

def filter_transcripts_by_fovs(transcripts_df, fovs):
    print("Filtering transcripts based on selected FOVs...")
    filtered_transcripts = transcripts_df[transcripts_df['fov_name'].isin(fovs)]
    return filtered_transcripts.compute()

def filter_boundaries_by_cells(file_path, cell_ids):
    print(f"Loading boundaries from {file_path} using Dask...")
    boundaries_df = dd.read_parquet(file_path)
    filtered_boundaries = boundaries_df[boundaries_df['cell_id'].isin(cell_ids)]
    return filtered_boundaries.compute()

def save_to_parquet(df, output_file):
    print(f"Saving data to {output_file}...")
    df.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}.")

def visualize_fovs_on_tiff(fovs_in_square, filtered_fovs_df, tiff_image_file, output_image_file, fov_column='fov_name', x_column='x', y_column='y', width_column='width', height_column='height'):
    print(f"Loading TIFF image from {tiff_image_file}...")
    tiff_image = tiff.imread(tiff_image_file)
    plt.figure(figsize=(10, 10))
    plt.imshow(tiff_image, cmap='gray')
    for _, row in filtered_fovs_df.iterrows():
        x, y, width, height = row[x_column], row[y_column], row[width_column], row[height_column]
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title("Selected FOVs over Morphology TIFF Image")
    plt.savefig(output_image_file)
    plt.show()

# Step 1: Get FOVs from the transcripts file for the middle square
square_size = 4  # Example: 4x4 FOVs
fovs_in_square, transcripts_df = find_fovs_in_square(transcripts_file, square_size)

# Step 2: Filter transcripts for the selected FOVs
filtered_transcripts = filter_transcripts_by_fovs(transcripts_df, fovs_in_square)

# Step 3: Get the cell_ids from the filtered transcripts
cell_ids_in_fovs = filtered_transcripts['cell_id'].unique()

# Step 4: Process and save filtered cell boundaries for the selected FOVs
cell_boundaries_df = filter_boundaries_by_cells(cell_boundaries_file, cell_ids_in_fovs)
save_to_parquet(cell_boundaries_df, output_dir / f"cell_boundaries.parquet")

# Step 5: Process and save filtered nuclei boundaries for the selected FOVs
nuclei_boundaries_df = filter_boundaries_by_cells(nuclei_file, cell_ids_in_fovs)
save_to_parquet(nuclei_boundaries_df, output_dir / f"nuclei_boundaries.parquet")

# Step 6: Process and save filtered transcripts for the selected FOVs
save_to_parquet(filtered_transcripts, output_dir / f"transcripts.parquet")

# Step 7: Visualize the selected FOVs as squares on top of the TIFF image and save the plot
visualize_fovs_on_tiff(fovs_in_square, filtered_transcripts, morphology_tiff_file, output_dir / "fovs_on_tiff.png")

print("Toy dataset generation and visualization complete!")
