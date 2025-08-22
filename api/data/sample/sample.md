# segger.data.sample

The `sample` module is the core of the Segger data processing framework, providing comprehensive classes for handling spatial transcriptomics data. This module contains the main classes that orchestrate the entire data processing pipeline from raw data to machine learning-ready graphs.


::: src.segger.data.sample
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_submodules: true
      merge_init_into_class: false
      members: true                 
      heading_level: 2           
      show_root_full_path: false


## Usage Examples

### Basic Data Loading

```python
from segger.data.sample import STSampleParquet

# Load a spatial transcriptomics sample
sample = STSampleParquet(
    base_dir="/path/to/xenium/data",
    n_workers=4,
    sample_type="xenium"
)

# Get sample information
print(f"Transcripts: {sample.n_transcripts}")
print(f"Spatial extents: {sample.extents}")
print(f"Feature names: {sample.transcripts_metadata['feature_names'][:5]}")
```

### Spatial Tiling and Processing

```python
# Save processed tiles
sample.save(
    data_dir="./processed_data",
    tile_size=1000,  # 1000 transcripts per tile
    k_bd=3,          # 3 boundary neighbors
    k_tx=5,          # 5 transcript neighbors
    dist_bd=15.0,    # 15 pixel boundary distance
    dist_tx=5.0,     # 5 pixel transcript distance
    frac=0.8,        # Process 80% of data
    val_prob=0.1,    # 10% validation
    test_prob=0.2    # 20% test
)
```

### In-Memory Dataset Processing

```python
from segger.data.sample import STInMemoryDataset

# Create dataset for a specific region
dataset = STInMemoryDataset(
    sample=sample,
    extents=region_polygon,
    margin=10
)

# Generate tiles
tiles = dataset._tile(
    width=100,    # 100 pixel width
    height=100    # 100 pixel height
)

print(f"Generated {len(tiles)} tiles")
```

### Individual Tile Processing

```python
from segger.data.sample import STTile

# Process individual tile
tile = STTile(dataset=dataset, extents=tile_polygon)

# Get tile data
transcripts = tile.transcripts
boundaries = tile.boundaries

# Convert to PyG format
pyg_data = tile.to_pyg_dataset(
    k_bd=3,
    dist_bd=15,
    k_tx=5,
    dist_tx=5,
    area=True,
    convexity=True,
    elongation=True,
    circularity=True
)

print(f"Tile UID: {tile.uid}")
print(f"Transcripts: {len(transcripts)}")
print(f"Boundaries: {len(boundaries)}")
```
