Certainly! Here's the full README content in Markdown format:

```markdown
# Segger Data Module

The `data` module provides tools and utilities for handling and processing spatial transcriptomics data, specifically designed to support Xenium and Merscope datasets. The module includes utilities for loading data, filtering transcripts, generating AnnData objects, and preparing data for downstream analysis.

## Features

- **Unified API**: A consistent interface for working with different spatial transcriptomics datasets (Xenium and Merscope).
- **Customizable Processing**: Methods for filtering transcripts, generating PyG graphs, and processing data tiles for machine learning models.
- **Extensible**: Easily extend the module to support additional spatial transcriptomics technologies.

## Installation

To install this module, simply clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

## Usage

### Loading and Filtering Data

#### Xenium Data

```python
from data.io import XeniumSample

# Load Xenium transcripts
xenium_sample = XeniumSample()
xenium_sample.load_transcripts(base_path='path/to/xenium/data', sample='sample_1')

# Optionally load boundaries (e.g., nuclei boundaries)
xenium_sample.load_boundaries(path='path/to/nucleus_boundaries.parquet')

# Filter transcripts based on quality value
filtered_transcripts = xenium_sample.filter_transcripts(xenium_sample.transcripts_df, min_qv=20.0)
```

#### Merscope Data

```python
from data.io import MerscopeSample

# Load Merscope transcripts
merscope_sample = MerscopeSample()
merscope_sample.load_transcripts(base_path='path/to/merscope/data', sample='sample_1', file_format='csv')

# Optionally load cell boundaries
merscope_sample.load_boundaries(path='path/to/cell_boundaries.parquet')

# Filter transcripts based on any specific criteria
filtered_transcripts = merscope_sample.filter_transcripts(merscope_sample.transcripts_df)
```

### Generating AnnData Objects

You can generate an `AnnData` object from the filtered transcripts, which is useful for downstream analysis.

```python
from data.utils import create_anndata

# Generate AnnData for Xenium
xenium_adata = create_anndata(
    df=filtered_transcripts,
    cell_id_col='cell_id'
)

# Generate AnnData for Merscope
merscope_adata = create_anndata(
    df=filtered_transcripts,
    cell_id_col='EntityID'
)
```

### Building PyG Graphs

For machine learning tasks, you can build PyG graphs from the data:

```python
from data.io import XeniumSample, MerscopeSample

# Xenium PyG Graph
xenium_data = xenium_sample.build_pyg_data_from_tile(
    boundaries_df=xenium_sample.boundaries_df,
    transcripts_df=filtered_transcripts,
)

# Merscope PyG Graph
merscope_data = merscope_sample.build_pyg_data_from_tile(
    boundaries_df=merscope_sample.boundaries_df,
    transcripts_df=filtered_transcripts,
)
```

### Saving Processed Tiles for Segger

You can save processed data tiles for training or testing machine learning models:

```python
# Save tiles for Xenium data
xenium_sample.save_dataset_for_segger(processed_dir='path/to/save/tiles')

# Save tiles for Merscope data
merscope_sample.save_dataset_for_segger(processed_dir='path/to/save/tiles')
```


## Methodology: Dataset Creation

### Overview

The dataset creation process involves the segmentation of spatial transcriptomics data into smaller, manageable tiles. These tiles are then used to build graph-based representations suitable for machine learning tasks. The process ensures that the spatial relationships between transcripts and their corresponding cells or nuclei are preserved.

### Step 1: Spatial Segmentation

Given a spatial region containing transcripts and cell/nucleus boundaries, we divide this region into smaller tiles of size \( x_{\text{size}} \times y_{\text{size}} \). Each tile is defined by its top-left corner coordinates \( (x_i, y_j) \).

For a grid defined by the spatial extents of the data, the tiles are created with a step size \( d_x \) and \( d_y \) along the \( x \)- and \( y \)-axes, respectively. The number of tiles along each axis is determined by:

\[
n_x = \left\lfloor \frac{x_{\text{max}} - x_{\text{min}}}{d_x} \right\rfloor, \quad n_y = \left\lfloor \frac{y_{\text{max}} - y_{\text{min}}}{d_y} \right\rfloor
\]

Where:
- \( x_{\text{min}}, y_{\text{min}} \) are the minimum coordinates.
- \( x_{\text{max}}, y_{\text{max}} \) are the maximum coordinates.

### Step 2: Transcript and Boundary Inclusion

For each tile defined by \( (x_i, y_j) \), we include all transcripts and boundaries (cells or nuclei) that fall within the tile's spatial bounds. Specifically, a transcript located at \( (x_t, y_t) \) is included if:

\[
x_i \leq x_t < x_i + x_{\text{size}}, \quad y_j \leq y_t < y_j + y_{\text{size}}
\]

Similarly, boundaries are included based on the vertices of their polygons. Optionally, a margin \( \text{margin}_x \) and \( \text{margin}_y \) can be added around each tile to include additional context:

\[
x_i - \text{margin}_x \leq x_t < x_i + x_{\text{size}} + \text{margin}_x, \quad y_j - \text{margin}_y \leq y_t < y_j + y_{\text{size}} + \text{margin}_y
\]

### Step 3: Graph Construction

For each tile, a graph \( G \) is constructed where:

- **Nodes**: The nodes \( V \) consist of transcripts and boundaries. Transcripts are represented by their spatial coordinates \( (x_t, y_t) \) and feature vectors \( \mathbf{f}_t \). Boundaries are represented by their centroid coordinates \( (x_b, y_b) \) and associated properties (e.g., area, convexity).

- **Edges**: Edges \( E \) are created between nodes based on spatial proximity. The proximity is determined using a method such as KD-Tree, FAISS, RAPIDS cuML, or cuGraph, and edges are created based on the \( k \)-nearest neighbors within a distance threshold \( d \):

\[
E = \{ (v_i, v_j) \mid \text{dist}(v_i, v_j) < d, \, v_i \in V, \, v_j \in V \}
\]

### Step 4: Label Computation (Optional)

If label computation is enabled, edges are labeled based on their relationships. For example, an edge between a transcript \( t \) and a boundary \( b \) might be labeled as belonging if \( t \) is within \( b \)'s boundary:

\[
\text{label}(t, b) = 
\begin{cases}
1 & \text{if } t \text{ belongs to } b \\
0 & \text{otherwise}
\end{cases}
\]

### Step 5: Dataset Splitting

The dataset is split into training, validation, and test sets based on probabilities \( p_{\text{train}}, p_{\text{val}}, p_{\text{test}} \), where:

\[
p_{\text{train}} + p_{\text{val}} + p_{\text{test}} = 1
\]

Each tile is randomly assigned to one of these sets based on the predefined probabilities.

### Output

The result of this process is a set of tiles, each containing a graph representation of the spatial transcriptomics data. These tiles are stored in the specified directories (`train_tiles`, `val_tiles`, `test_tiles`) and are ready for use in machine learning pipelines.
