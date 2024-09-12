# 🍳 Welcome to segger 

**segger** is a cutting-edge tool for **cell segmentation** in **single-molecule spatial omics** datasets. By leveraging **graph neural networks (GNNs)** and heterogeneous graphs, segger offers unmatched accuracy and scalability.

# How segger Works

![Segger Model](docs/images/Segger_model_08_2024.png)

---

# Quick Links

- 💾 **[Installation Guide](https://elihei2.github.io/segger_dev/installation/)**  
  Get started with installing segger on your machine.

- 📖 **[User Guide](https://elihei2.github.io/segger_dev/user_guide/)**  
  Learn how to use segger for cell segmentation tasks.

- 💻 **[Command-Line Interface (CLI)](https://elihei2.github.io/segger_dev/cli/)**  
  Explore the CLI options for working with segger.

- 📚 **[API Reference](https://elihei2.github.io/segger_dev/api/)**  
  Dive into the detailed API documentation for advanced usage.

---

# Why segger?

- ⚙️ **Highly parallelizable** – Optimized for multi-GPU environments
- ⚡ **Fast and efficient** – Trains in a fraction of the time compared to alternatives
- 🔄 **Transfer learning** – Easily adaptable to new datasets and technologies

### Challenges in Segmentation

Spatial omics segmentation faces issues like:

- **Over/Under-segmentation**
- **Transcript contamination**
- **Scalability limitations**

segger tackles these with a **graph-based approach**, achieving superior segmentation accuracy.

---
## Installation Options

Choose the installation method that best suits your needs.

### Micromamba Installation

```bash
micromamba create -n segger-rapids --channel-priority 1 \
    -c rapidsai -c conda-forge -c nvidia -c pytorch -c pyg \
    rapids=24.08 python=3.* 'cuda-version>=11.4,<=11.8' jupyterlab \
    'pytorch=*=*cuda*' 'pyg=*=*cu118' pyg-lib pytorch-sparse
micromamba install -n segger-rapids --channel-priority 1 --file mamba_environment.yml
micromamba run -n segger-rapids pip install --no-deps ./
```

### GitHub Installation

```bash
git clone https://github.com/EliHei2/segger_dev.git
cd segger_dev
pip install .
```

---

# Powered by

- ⚡ **PyTorch Lightning & PyTorch Geometric**: Enables fast, efficient graph neural network (GNN) implementation for heterogeneous graphs.
- ⚙️ **Dask**: Scalable parallel processing and distributed task scheduling, ideal for handling large transcriptomic datasets.
- 🗺️ **Shapely & Geopandas**: Utilized for spatial operations such as polygon creation, scaling, and spatial relationship computations.
- 🖥️ **RAPIDS**: Provides GPU-accelerated computation for tasks like k-nearest neighbors (KNN) graph construction.
- 📊 **AnnData & Scanpy**: Efficient processing for single-cell datasets.
- 📐 **SciPy**: Facilitates spatial graph construction, including distance metrics and convex hull calculations for transcript clustering.

---

# Contributions

segger is **open-source** and welcomes contributions. Join us in advancing spatial omics segmentation!

- 🛠️ **Source Code**  
  [GitHub](https://github.com/EliHei2/segger_dev)

- 🐞 **Bug Tracker**  
  [Report Issues](https://github.com/EliHei2/segger_dev/issues)

- 📚 **Full Documentation**  
  [API Reference](https://elihei2.github.io/segger_dev/api/)
