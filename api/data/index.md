# segger.data

The `segger.data` module in **Segger** is designed to facilitate data preparation for spatial transcriptomics datasets, focusing on **Xenium** and **Merscope**. It provides a unified, scalable interface for handling large datasets, performing boundary and transcript filtering, and preparing data for graph-based deep learning models.
## Submodules

- [Constants](constants/index.md): Contains predefined constants for spatial data processing.
- [IO](io/index.md): Provides utilities for input/output operations, supporting various data formats.
- [Utils](utils/index.md): Miscellaneous utility functions for processing and analysis.

## Key Features

- **Lazy Loading**: Efficiently handles large datasets using **Dask**, avoiding memory bottlenecks.
- **Flexible Tiling**: Spatially segments datasets into tiles, optimizing for parallel processing.
- **Graph Construction**: Converts spatial data into **PyTorch Geometric (PyG)** graph formats for **GNN**-based models.
- **Boundary Handling**: Processes spatial polygons and computes transcript overlaps.
- **Dataset-agnostic API**: Unified API for multiple spatial transcriptomics technologies, such as **Xenium** and **Merscope**.



## API Documentation

::: segger.data

