# Welcome to segger 🥚

**segger** is a cutting-edge tool for **cell segmentation** in **single-molecule spatial omics** datasets. By leveraging **graph neural networks (GNNs)** and heterogeneous graphs, segger offers unmatched accuracy and scalability.

> **Note**:  
> For installation instructions, visit the [Installation Guide](installation.md).

---

# Quick Links

- **[Installation Guide](installation.md)**: Get started with installing segger on your machine.
- **[User Guide](user_guide/index.md)**: Learn how to use segger for cell segmentation tasks.
- **[Command-Line Interface (CLI)](cli.md)**: Explore the CLI options for working with segger.
- **[API reference](api/index.md)**: Dive into the detailed API documentation for advanced usage.

---

# Why segger?

### **Challenges in Segmentation**

Spatial omics segmentation faces issues like:

- **Over/Under-segmentation**
- **Transcript contamination**
- **Scalability limitations**

segger tackles these with a **graph-based approach**, achieving superior segmentation accuracy.

---

# How segger Works

### **Heterogeneous Graph Structure**

segger models **nuclei and transcript relationships** using:

- **Nodes** for nuclei and transcripts
- **Edges** connecting spatially related elements
- **GNN** to propagate information and refine cell borders

**Nucleus-free transcripts**: segger can effectively handle **nucleus-free cells**, improving transcript assignment in complex tissue environments.

---

# Key Features

### **What makes segger unique?**

- **Highly parallelizable** 🗂️ – Optimized for multi-GPU environments
- **Fast and efficient** ⚡– Trains in a fraction of the time compared to alternatives
- **Transfer learning** 🔄– Easily adaptable to new datasets and technologies

---

# Powered by

- **PyTorch Lightning & PyTorch Geometric**: Enables fast, efficient graph neural network (GNN) implementation for heterogeneous graphs.
- **Dask**: Enables scalable parallel processing and distributed task scheduling, making it ideal for handling large transcriptomic datasets and accelerating data processing workflows.
- **RAPIDS**: Provides GPU-accelerated computation for tasks like k-nearest neighbors (KNN) graph construction.
- **AnnData & Scanpy**: Handles transcriptomic data integration and efficient data processing for large-scale datasets.
- **SciPy**: Facilitates spatial graph construction, including distance metrics and convex hull calculations for transcript clustering.

---

# Contributions

- **Source Code**: [GitHub](https://github.com/EliHei2/segger_dev)
- **Bug Tracker**: [Report Issues](https://github.com/EliHei2/segger_dev/issues)
- **Full Documentation**: [API Reference](api/index.md)

segger is **open-source** and welcomes contributions. Join us in advancing spatial omics segmentation!
