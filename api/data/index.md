# segger.data

The `segger.data` module in **Segger** is a comprehensive data processing and management system designed specifically for spatial transcriptomics datasets. It provides a unified, scalable interface for handling large-scale spatial transcriptomics data from technologies such as **Xenium** and **Merscope**, with a focus on preparing data for graph-based deep learning models.


- [Sample](sample.md): Core sample handling, tiling, and data management classes
- [PyG Dataset](pyg_dataset.md): PyTorch Geometric dataset integration and utilities
- [Utils](_utils.md): Core utility functions for data processing, filtering, and analysis
- [Transcript Embedding](transcript_embedding.md): Transcript feature encoding and embedding utilities
- [NDTree](ndtree.md): Spatial partitioning and load balancing utilities


## Module Overview

The `segger.data` package is organized into several key modules, each serving a specific purpose in the spatial transcriptomics data processing pipeline.

### [Sample Module](sample.md)
The main data processing module containing the core classes for handling spatial transcriptomics data.


**Main Functions:**

- Data loading and validation
- Spatial tiling and partitioning
- Graph construction and feature engineering
- Parallel processing coordination

**Key Classes:**

- `STSampleParquet`: Main orchestrator for data loading and processing
- `STInMemoryDataset`: In-memory dataset with spatial indexing
- `STTile`: Individual spatial tile processing


### [Utils Module](_utils.md)
Core utility functions for data processing, filtering, and analysis.

**Key Functions:**

- `get_xy_extents()`: Extract spatial extents from parquet files
- `read_parquet_region()`: Read specific spatial regions from parquet files
- `filter_transcripts()`: Filter transcripts by quality and gene type
- `filter_boundaries()`: Filter boundaries by spatial criteria
- `load_settings()`: Load technology-specific configurations
- `find_markers()`: Identify marker genes for cell types

### [PyG Dataset Module](pyg_dataset.md)
PyTorch Geometric dataset integration for machine learning workflows.

- Automatic tile discovery and loading
- PyTorch Lightning integration
- Built-in data validation
- Efficient memory management

**Key Classes:**

- `STPyGDataset`: PyTorch Geometric dataset wrapper


### [Transcript Embedding Module](transcript_embedding.md)
Utilities for encoding transcript features into numerical representations.


### [NDTree Module](ndtree.md)
Spatial partitioning and load balancing utilities.

**Key Classes:**

- `NDTree`: N-dimensional spatial partitioning
- `innernode`: Internal tree node structure

**Features:**

- Efficient spatial data partitioning
- Load balancing for parallel processing
- Memory-optimized data structures
- Configurable region sizing

## Configuration and Settings

### [Settings Directory](../settings/)
Technology-specific configuration files for different spatial transcriptomics platforms.

**Available Platforms:**

- Xenium
- Merscope
- CosMx
- Xenium v2 (segmentation kit)

**Configuration Options:**

- Column mappings
- Quality thresholds
- Spatial parameters
- Platform-specific defaults

## Data Flow Architecture

```
Raw Data (Parquet) → STSampleParquet → Spatial Tiling → STInMemoryDataset → STTile → PyG Graph
     ↓                      ↓              ↓                ↓              ↓
Metadata Extraction    Region Division   Data Filtering   Tile Creation   Graph Construction
     ↓                      ↓              ↓                ↓              ↓
Settings Loading      Load Balancing    Spatial Indexing   Feature Comp.   Edge Generation
```

## Class Hierarchy

```
STSampleParquet (Main Orchestrator)
├── STInMemoryDataset (Region Processing)
│   └── STTile (Individual Tile Processing)
├── TranscriptEmbedding (Feature Encoding)
└── NDTree (Spatial Partitioning)

STPyGDataset (ML Integration)
└── PyTorch Geometric Integration

BackendHandler (Experimental)
└── Multi-backend Support
```

## Usage 

```python
from segger.data.sample import STSampleParquet

# Load and process data
sample = STSampleParquet(base_dir="path/to/data", n_workers=4)
sample.save(data_dir="./processed", tile_size=1000)
```




