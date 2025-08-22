# Data Module API Reference

This page provides a comprehensive reference to all the classes, functions, and modules in the `segger.data` package.

## Module Overview

The `segger.data` package is organized into several key modules, each serving a specific purpose in the spatial transcriptomics data processing pipeline.

## Core Modules

### [Sample Module](sample.md)
The main data processing module containing the core classes for handling spatial transcriptomics data.

**Key Classes:**
- `STSampleParquet`: Main orchestrator for data loading and processing
- `STInMemoryDataset`: In-memory dataset with spatial indexing
- `STTile`: Individual spatial tile processing

**Main Functions:**
- Data loading and validation
- Spatial tiling and partitioning
- Graph construction and feature engineering
- Parallel processing coordination

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

**Key Classes:**
- `STPyGDataset`: PyTorch Geometric dataset wrapper

**Features:**
- Automatic tile discovery and loading
- PyTorch Lightning integration
- Built-in data validation
- Efficient memory management

### [Transcript Embedding Module](transcript_embedding.md)
Utilities for encoding transcript features into numerical representations.

**Key Classes:**
- `TranscriptEmbedding`: Transcript feature encoding and embedding

**Features:**
- Index-based and weighted embeddings
- PyTorch module integration
- Input validation and error handling
- Support for custom embedding strategies

### [Experimental Module](experimental.md)
Experimental features and multi-backend support.

**Key Classes:**
- `BackendHandler`: Multi-backend DataFrame support

**Supported Backends:**
- pandas: Standard data processing
- dask: Parallel processing
- cudf: GPU acceleration
- dask_cudf: Distributed GPU processing

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
- Xenium v2

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

## Common Usage Patterns

### 1. Basic Data Processing

```python
from segger.data.sample import STSampleParquet

# Load and process data
sample = STSampleParquet(base_dir="path/to/data", n_workers=4)
sample.save(data_dir="./processed", tile_size=1000)
```

### 2. Custom Processing

```python
from segger.data.sample import STInMemoryDataset, STTile

# Process specific regions
dataset = STInMemoryDataset(sample=sample, extents=region)
tiles = dataset._tile(width=100, height=100)

# Process individual tiles
for tile in tiles:
    pyg_data = tile.to_pyg_dataset()
    # Continue processing...
```

### 3. Machine Learning Integration

```python
from segger.data.pyg_dataset import STPyGDataset

# Load processed data
dataset = STPyGDataset(root="./processed_data")

# Use in training
for data in dataset:
    # Train your model...
```

## Performance Considerations

### Memory Management
- **Lazy Loading**: Data loaded only when needed
- **Spatial Filtering**: Only relevant data in memory
- **Intelligent Caching**: Frequently accessed data cached
- **Parallel Processing**: Multiple workers process different regions

### Optimization Strategies
- **Tile Size Selection**: Balance memory usage and processing efficiency
- **Worker Configuration**: Match worker count to available resources
- **Spatial Indexing**: KDTree-based fast spatial queries
- **Quality Filtering**: Early filtering to reduce processing overhead

## Error Handling

The package includes comprehensive error handling:

- **File Validation**: Checks for valid data files and formats
- **Data Quality**: Validates transcript and boundary data
- **Memory Management**: Handles out-of-memory situations gracefully
- **Worker Coordination**: Manages worker failures and recovery

## Best Practices

### Data Organization
- Use consistent directory structures
- Include metadata in file names
- Validate data quality before processing
- Monitor memory usage during processing

### Performance Tuning
- Choose appropriate tile sizes for your hardware
- Balance worker count with available resources
- Use debug mode for initial setup and troubleshooting
- Monitor processing times and adjust parameters

### Quality Control
- Set appropriate quality thresholds
- Filter unwanted genes early in the pipeline
- Validate spatial relationships
- Check output data consistency

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce tile size or number of workers
2. **Slow Processing**: Check spatial indexing and parallelization
3. **Data Quality Issues**: Verify input data format and quality metrics
4. **Worker Failures**: Check worker configuration and resource availability

### Debug Tips
- Use `save_debug()` for detailed processing information
- Monitor memory usage during processing
- Check tile generation parameters
- Validate spatial relationships in output data

## Future Developments

Planned improvements include:

- **Additional Platforms**: Support for more spatial transcriptomics technologies
- **Enhanced Tiling**: More sophisticated spatial partitioning strategies
- **Real-time Processing**: Streaming data processing capabilities
- **Cloud Integration**: Support for cloud-based data processing
- **Advanced Features**: More sophisticated feature engineering and graph construction

## Contributing

Contributions are welcome! Areas for improvement include:

- **New Data Formats**: Support for additional spatial transcriptomics platforms
- **Performance Optimization**: Better algorithms and data structures
- **Documentation**: Examples, tutorials, and best practices
- **Testing**: Comprehensive test coverage and validation

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **geopandas**: Spatial data handling
- **shapely**: Geometric operations
- **pyarrow**: Parquet file handling
- **scipy**: Spatial data structures

### ML Dependencies
- **torch**: PyTorch integration
- **torch_geometric**: Graph neural network support
- **pytorch_lightning**: Training framework integration

### Optional Dependencies
- **dask**: Parallel processing support
- **cudf**: GPU acceleration
- **scanpy**: Single-cell analysis
- **anndata**: Annotated data structures
