# segger - Data Preparation for Cell Segmentation

The `segger` package provides a comprehensive data preparation module for handling and processing spatial transcriptomics data, specifically designed to support **Xenium** and **Merscope** datasets. This module facilitates the creation of datasets for cell segmentation and subsequent graph-based deep learning tasks by leveraging scalable and efficient processing tools.

## Module Overview

The data preparation module offers the following key functionalities:

1. **Lazy Loading of Large Datasets**: Utilizes **Dask** to handle large-scale transcriptomics and boundary datasets efficiently, avoiding memory bottlenecks.
2. **Flexible and Efficient Tiling**: Divides datasets into spatial tiles, essential for localized graph-based models and parallel processing.
3. **Graph Construction**: Converts spatial data into graph formats using **PyTorch Geometric (PyG)**, enabling the application of graph neural networks (GNNs).
4. **Boundary Processing**: Handles polygons, performs spatial geometrical calculations, and checks transcript overlaps with boundaries.
5. **Data Filtering**: Filters transcripts based on quality metrics and dataset-specific criteria to ensure data integrity and relevance.
6. **Unified API for Multiple Datasets**: Provides a consistent interface for working with both Xenium and Merscope datasets, facilitating extensibility to additional spatial transcriptomics technologies.

## Key Technologies

- **Dask**: Facilitates parallel and lazy data processing, enabling scalable handling of large datasets.
- **PyTorch Geometric (PyG)**: Enables the construction of graph-based data representations suitable for GNNs.
- **Shapely & Geopandas**: Utilized for spatial operations such as polygon creation, scaling, and spatial relationship computations.
- **Dask-Geopandas**: Extends Geopandas for parallel processing of geospatial data, enhancing scalability.

## Core Classes

### 1. `SpatialTranscriptomicsSample` (Abstract Base Class)

This abstract class defines the foundational structure for managing spatial transcriptomics datasets. It provides essential methods for:

- **Loading Data**: Scalable loading of transcript and boundary data using Dask.
- **Filtering Transcripts**: Applying quality-based or dataset-specific filtering criteria.
- **Spatial Relationships**: Computing overlaps and spatial relationships between transcripts and boundaries.
- **Tiling**: Dividing datasets into smaller spatial tiles for localized processing.
- **Graph Preparation**: Converting data tiles into `PyTorch Geometric` graph structures.

#### Key Methods:

- **`load_transcripts()`**: Loads transcriptomic data from Parquet files, applies quality filtering, and incorporates additional gene embeddings.
- **`load_boundaries()`**: Loads boundary data (e.g., cell or nucleus boundaries) from Parquet files.
- **`get_tile_data()`**: Retrieves transcriptomic and boundary data within specified spatial bounds.
- **`generate_and_scale_polygons()`**: Creates and scales polygon representations of boundaries for spatial computations.
- **`compute_transcript_overlap_with_boundaries()`**: Determines the association of transcripts with boundary polygons.
- **`build_pyg_data_from_tile()`**: Converts tile-specific data into `HeteroData` objects suitable for PyG models.

### 2. `XeniumSample` and `MerscopeSample` (Child Classes)

These classes inherit from `SpatialTranscriptomicsSample` and implement dataset-specific processing logic:

- **`XeniumSample`**: Tailored for **Xenium** datasets, it includes specific filtering rules to exclude unwanted transcripts based on naming patterns (e.g., `NegControlProbe_`, `BLANK_`).
- **`MerscopeSample`**: Designed for **Merscope** datasets, allowing for custom filtering and processing logic as needed.

## Workflow

The dataset creation and processing workflow involves several key steps, each ensuring that the spatial transcriptomics data is appropriately prepared for downstream machine learning tasks.

### Step 1: Data Loading and Filtering

- **Transcriptomic Data**: Loaded lazily using Dask to handle large datasets efficiently. Custom filtering rules specific to the dataset (Xenium or Merscope) are applied to ensure data quality.
- **Boundary Data**: Loaded similarly using Dask, representing spatial structures such as cell or nucleus boundaries.

### Step 2: Tiling

- **Spatial Segmentation**: The dataset is divided into smaller, manageable tiles of size $$x_{\text{size}} \times y_{\text{size}}$$, defined by their top-left corner coordinates $$(x_i, y_j)$$.

$$
n_x = \left\lfloor \frac{x_{\text{max}} - x_{\text{min}}}{d_x} \right\rfloor, \quad n_y = \left\lfloor \frac{y_{\text{max}} - y_{\text{min}}}{d_y} \right\rfloor
$$

Where:

- $$x_{\text{min}}, y_{\text{min}}$$: Minimum spatial coordinates.
- $$x_{\text{max}}, y_{\text{max}}$$: Maximum spatial coordinates.
- $$d_x, d_y$$: Step sizes along the $$x$$- and $$y$$-axes, respectively.

- **Transcript and Boundary Inclusion**: For each tile, transcripts and boundaries within the spatial bounds (with optional margins) are included:

$$
x_i - \text{margin}_x \leq x_t < x_i + x_{\text{size}} + \text{margin}_x, \quad y_j - \text{margin}_y \leq y_t < y_j + y_{\text{size}} + \text{margin}_y
$$

Where:

- $$x_t, y_t$$: Transcript coordinates.
- $$\text{margin}_x, \text{margin}_y$$: Optional margins to include contextual data.

### Step 3: Graph Construction

For each tile, a graph $$G$$ is constructed with:

- **Nodes ($$V$$)**:

  - **Transcripts**: Represented by their spatial coordinates $$(x_t, y_t)$$ and feature vectors $$\mathbf{f}_t$$.
  - **Boundaries**: Represented by centroid coordinates $$(x_b, y_b)$$ and associated properties (e.g., area).

- **Edges ($$E$$)**:
  - Created based on spatial proximity using methods like KD-Tree or FAISS.
  - Defined by a distance threshold $$d$$ and the number of nearest neighbors $$k$$:

$$
E = \{ (v_i, v_j) \mid \text{dist}(v_i, v_j) < d, \, v_i \in V, \, v_j \in V \}
$$

### Step 4: Label Computation (Optional)

If enabled, edges can be labeled based on relationships, such as whether a transcript belongs to a boundary:

$$
\text{label}(t, b) =
\begin{cases}
1 & \text{if } t \text{ belongs to } b \\
0 & \text{otherwise}
\end{cases}
$$

### Step 5: Dataset Splitting

The dataset is partitioned into training, validation, and test sets based on predefined probabilities $$p_{\text{train}}, p_{\text{val}}, p_{\text{test}}$$:

$$
p_{\text{train}} + p_{\text{val}} + p_{\text{test}} = 1
$$

Each tile is randomly assigned to one of these sets according to the specified probabilities.

### Output

The final output consists of a set of tiles, each containing a graph representation of the spatial transcriptomics data. These tiles are stored in designated directories (`train_tiles`, `val_tiles`, `test_tiles`) and are ready for integration into machine learning pipelines.

## Example Usage

Below are examples demonstrating how to utilize the `segger` data preparation module for both Xenium and Merscope datasets.

### Xenium Data

```python
from segger.data.io import XeniumSample
from pathlib import Path
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import os

xenium_data_dir = Path("./data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1")
segger_data_dir = Path("./data_tidy/pyg_datasets/bc_embedding_0919")
models_dir = Path("./models/bc_embedding_0919")

scRNAseq_path = (
    "/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad"
)

scRNAseq = sc.read(scRNAseq_path)

sc.pp.subsample(scRNAseq, 0.1)

# Step 1: Calculate the gene cell type abundance embedding
celltype_column = "celltype_minor"
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
    scRNAseq, celltype_column
)

# Setup Xenium sample to create dataset
xs = XeniumSample(verbose=False, embedding_df=gene_celltype_abundance_embedding)
xs.set_file_paths(
    transcripts_path=xenium_data_dir / "transcripts.parquet",
    boundaries_path=xenium_data_dir / "nucleus_boundaries.parquet",
)
xs.set_metadata()

# Set the embedding to "cell_type_abundance" to use it in further processing
xenium_sample.set_embedding("cell_type_abundance")

# Load nuclei data to define boundaries
nuclei_path = raw_data_dir / sample_tag / "nucleus_boundaries.parquet"
xenium_sample.load_boundaries(path=nuclei_path, file_format="parquet")

# Build PyTorch Geometric (PyG) data from a tile of the dataset
tile_pyg_data = xenium_sample.build_pyg_data_from_tile(
    boundaries_df=xenium_sample.boundaries_df,
    transcripts_df=xenium_sample.transcripts_df,
    r_tx=20,
    k_tx=20,
    use_precomputed=False,
    workers=1,
)


try:
    xs.save_dataset_for_segger(
        processed_dir=segger_data_dir,
        x_size=400,
        y_size=400,
        d_x=350,
        d_y=350,
        margin_x=20,
        margin_y=20,
        compute_labels=True,  # Set to True if you need to compute labels
        r_tx=5,
        k_tx=10,
        val_prob=0.4,
        test_prob=0.1,
        num_workers=6,
    )
except AssertionError as err:
    print(f"Dataset already exists at {segger_data_dir}")
```

### Merscope Data

```python
from segger.data import MerscopeSample
from pathlib import Path

# Set up the file paths
raw_data_dir = Path("data_raw/merscope/")
processed_data_dir = Path("data_tidy/pyg_datasets")
sample_tag = "Merscope_Sample_1"

# Create a MerscopeSample instance for spatial transcriptomics processing
merscope_sample = MerscopeSample()

# Load transcripts from a CSV file
merscope_sample.load_transcripts(
    base_path=raw_data_dir,
    sample=sample_tag,
    transcripts_filename="transcripts.csv",
    file_format="csv",
)

# Optionally load cell boundaries
cell_boundaries_path = raw_data_dir / sample_tag / "cell_boundaries.parquet"
merscope_sample.load_boundaries(path=cell_boundaries_path, file_format="parquet")

# Filter transcripts based on specific criteria
filtered_transcripts = merscope_sample.filter_transcripts(
    merscope_sample.transcripts_df
)

# Build PyTorch Geometric (PyG) data from a tile of the dataset
tile_pyg_data = merscope_sample.build_pyg_data_from_tile(
    boundaries_df=merscope_sample.boundaries_df,
    transcripts_df=filtered_transcripts,
    r_tx=15,
    k_tx=15,
    use_precomputed=True,
    workers=2,
)

# Save dataset in processed format for segmentation
merscope_sample.save_dataset_for_segger(
    processed_dir=processed_data_dir / "embedding",
    x_size=360,
    y_size=360,
    d_x=180,
    d_y=180,
    margin_x=10,
    margin_y=10,
    compute_labels=True,
    r_tx=5,
    k_tx=5,
    val_prob=0.1,
    test_prob=0.2,
    neg_sampling_ratio_approx=3,
    sampling_rate=1,
    num_workers=2,
)
```
