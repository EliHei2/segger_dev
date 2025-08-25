# Data Preparation for `segger`

The `segger` package provides a streamlined, settings-driven pipeline to transform raw spatial transcriptomics outputs (e.g., Xenium, Merscope) into graph tiles ready for model training and evaluation.


!!! note
    Currently, `segger` supports **Xenium** and **Merscope** datasets via technology-specific settings.


## Steps

The data preparation pipeline includes:

1. **Settings-driven I/O**: Uses a `sample_type` (e.g., `xenium`, `merscope`) to resolve input file names and column mappings.
2. **Lazy Loading + Filtering**: Efficiently reads Parquet in spatial regions and filters transcripts/boundaries.
3. **Tiling**: Partitions the whole slide into spatial tiles (fixed size or balanced by transcript count).
4. **Graph Construction**: Builds PyTorch Geometric `HeteroData` with typed nodes/edges and labels for link prediction.
5. **Splitting**: Writes tiles into `train`, `val`, and `test` subsets.


!!! note "Key Technologies"
      - **PyTorch Geometric (PyG)**: Heterogeneous graphs for GNNs.
      - **Shapely & GeoPandas**: Geometry operations (polygons, centroids, areas).
      - **PyArrow Parquet**: Efficient I/O with schema-aware reads.


## Core Components

### 1. `STSampleParquet`

Settings-based entry point for preparing a sample into graph tiles.

- **Constructor**: resolves input file paths and metadata using `sample_type` settings (e.g., file names, column names, quality fields). Ensures transcript IDs exist.
- **Embeddings**: optionally accepts a `weights` DataFrame (index: gene names; columns: embedding dims) to encode transcript features.
- **Saving**: orchestrates region partitioning, tiling, PyG graph construction, negative sampling, and dataset splitting.

#### Key Params
- `base_dir`: folder with required Parquet files (transcripts and boundaries).
- `sample_type`: one of `xenium`, `merscope` (determines settings such as file names, columns, nuclear flags, scale factors).
- `weights`: optional `pd.DataFrame` of gene embeddings used by `TranscriptEmbedding`.
- `scale_factor`: optional override for boundary scaling used during spatial queries.


### 2. `STInMemoryDataset`

Internal helper that loads filtered transcripts/boundaries for a region, pre-builds a KDTree, and generates tile bounds (fixed-size or balanced by count).


### 3. `STTile`

Per-tile builder that assembles a `HeteroData` graph:
- Nodes: `tx` (transcripts) with `pos`, `x` (features); `bd` (boundaries) with `pos`, `x` (polygon properties).
- Edges:
  - `('tx','neighbors','tx')`: transcript proximity (KDTree-based; `k_tx`, `dist_tx`).
  - `('tx','neighbors','bd')`: transcript-to-boundary proximity for receptive field construction.
  - `('tx','belongs','bd')`: positive labels from nuclear overlap or provided assignment; negative sampling performed from receptive-field candidates.


## Workflow

### Step 1: Initialize sample from settings

- Provide `base_dir` containing technology outputs.
- Pick `sample_type` to resolve filenames/columns.
- Optionally provide `weights` for transcript embeddings.

### Step 2: Region partitioning and tiling

- If multiple workers are set, extents are split into balanced regions (ND-tree over boundaries).
- Tiles are created either by fixed width/height or by a target `tile_size` (balanced by transcript count).

### Step 3: Graph construction per tile

- Build `HeteroData` with transcript (`tx`) and boundary (`bd`) nodes.
- Add proximity edges and `belongs` labels (positives + sampled negatives).

### Step 4: Splitting and saving

- Tiles are written to `<data_dir>/{train_tiles,val_tiles,test_tiles}/processed/*.pt` according to `val_prob`/`test_prob`.


## Output

- A directory structure with train/val/test tiles in PyG `HeteroData` format ready for the Segger model and `STPyGDataset`.

```
<data_dir>/
  train_tiles/
    processed/
      tiles_x=..._y=..._w=..._h=....pt
  val_tiles/
    processed/
      ...
  test_tiles/
    processed/
      ...
```


## Example Usage

### Xenium (with optional scRNA-seq-derived embeddings)

```python
from pathlib import Path
import pandas as pd

# Optional: provide transcript embeddings (rows: genes, cols: embedding dims)
# For example, cell-type abundance embeddings indexed by gene name
# weights = pd.DataFrame(..., index=gene_names)
weights = None  # set to a DataFrame if available

from segger.data.sample import STSampleParquet

base_dir = Path("/path/to/xenium_output")
data_dir = Path("/path/to/processed_tiles")

sample = STSampleParquet(
    base_dir=base_dir,
    sample_type="xenium",
    n_workers=4,            # controls parallel tiling across regions
    # weights=weights,        # optional transcript embeddings
    scale_factor=1.0,       # optional override (geometry scaling)
)

# Save tiles (choose either tile_size OR tile_width+tile_height)
sample.save(
    data_dir=data_dir,
    # Receptive fields (neighbors)
    k_bd=3,        # nearest boundaries per transcript
    dist_bd=15.0,  # max distance for tx->bd neighbors (µm-equivalent)
    k_tx=20,       # nearest transcripts per transcript
    dist_tx=5.0,   # max distance for tx->tx neighbors
    # Optional broader receptive fields for mutually exclusive genes (if used)
    # Tiling
    tile_size=50000,   # alternative: tile_width=..., tile_height=...
    # Sampling/splitting
    neg_sampling_ratio=5.0,
    frac=1.0,
    val_prob=0.1,
    test_prob=0.2,
)
```

### Merscope (fixed-size tiling)

```python
from pathlib import Path
from segger.data.sample import STSampleParquet

base_dir = Path("/path/to/merscope_output")
data_dir = Path("/path/to/processed_tiles")

sample = STSampleParquet(
    base_dir=base_dir,
    sample_type="merscope",
    n_workers=2,
)

sample.save(
    data_dir=data_dir,
    # Nearest neighbors
    k_bd=3,
    dist_bd=15.0,
    k_tx=15,
    dist_tx=5.0,
    # Fixed-size tiling in sample units
    tile_width=300,
    tile_height=300,
    # Splits
    neg_sampling_ratio=3.0,
    val_prob=0.1,
    test_prob=0.2,
)
```

### Debug mode (step-by-step logging)

```python
sample.save_debug(
    data_dir=data_dir,
    k_bd=3,
    dist_bd=15.0,
    k_tx=20,
    dist_tx=5.0,
    tile_width=300,
    tile_height=300,
    neg_sampling_ratio=5.0,
    frac=1.0,
    val_prob=0.1,
    test_prob=0.2,
)
```


## Notes and Recommendations

- **Settings and columns**: Filenames and columns for transcripts/boundaries are resolved via `sample_type` settings. See `segger.data._settings/*` for details.
- **Transcript IDs**: The constructor ensures an ID column exists in transcripts; if missing, it is added deterministically.
- **Quality filtering**: Uses settings-defined columns (e.g., QV) and filter substrings. Genes absent from provided `weights` will be auto-added to filter substrings to avoid OOV embeddings.
- **Neighbors**: Set `k_tx/dist_tx` based on typical nuclear radii and transcript densities; `k_bd/dist_bd` controls candidate boundaries per transcript.
- **Splits**: Tiles with no `('tx','belongs','bd')` edges are automatically placed in `test_tiles`.
- **Embeddings**: If no `weights` are provided, transcripts fall back to token/ID-based embeddings.

## Using scRNA-seq for embeddings and mutually exclusive genes

You can leverage scRNA-seq data both to create transcript embeddings (weights) and to identify mutually exclusive gene pairs that guide repulsive/attractive transcript edges.

### 1) Compute transcript embeddings (weights) from scRNA-seq

```python
import scanpy as sc
from segger.data._utils import calculate_gene_celltype_abundance_embedding

# Load a reference AnnData
adata = sc.read("/path/to/reference_scrnaseq.h5ad")
sc.pp.subsample(adata, 0.25)        # optional downsampling
adata.var_names_make_unique()
sc.pp.log1p(adata)
sc.pp.normalize_total(adata)

# Column in adata.obs with cell-type annotations
celltype_column = "celltype_minor"

# Compute gene x cell-type abundance matrix (DataFrame indexed by gene names)
weights = calculate_gene_celltype_abundance_embedding(
    adata,
    celltype_column,
)

# Pass weights to STSampleParquet to encode transcript features
from segger.data.sample import STSampleParquet
sample = STSampleParquet(
    base_dir="/path/to/technology_output",
    sample_type="xenium",      # or "merscope"
    n_workers=4,
    weights=weights,
)
```

### 2) [OPTIONAL] Identify mutually exclusive genes from scRNA-seq

```python
from segger.data._utils import find_markers, find_mutually_exclusive_genes

# Optionally restrict to genes present in the sample
genes = list(set(adata.var_names) & set(sample.transcripts_metadata["feature_names"]))
adata_sub = adata[:, genes]

# Find cell-type markers (tune thresholds as needed)
markers = find_markers(
    adata_sub,
    cell_type_column=celltype_column,
    pos_percentile=90,
    neg_percentile=20,
    percentage=20,
)

# Compute mutually exclusive gene pairs using markers
exclusive_gene_pairs = find_mutually_exclusive_genes(
    adata=adata,
    markers=markers,
    cell_type_column=celltype_column,
)
```

### 3) Save tiles with both weights and mutually exclusive genes

```python
sample.save(
    data_dir="/path/to/processed_tiles",
    # Nearest-neighbor receptive fields
    k_bd=3, dist_bd=15.0,
    k_tx=20, dist_tx=5.0,
    # Optional broader receptive fields used for mutually exclusive genes
    k_tx_ex=100, dist_tx_ex=20.0,
    # Tiling and splits
    tile_size=50_000,
    neg_sampling_ratio=5.0,
    val_prob=0.1, test_prob=0.2,
    # Use mutually exclusive pairs to add repulsive/attractive tx-tx labels
    mutually_exclusive_genes=exclusive_gene_pairs,
)
```
