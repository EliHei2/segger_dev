# 🍳 Welcome to segger!

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EliHei2/segger_dev/main.svg)](https://results.pre-commit.ci/latest/github/EliHei2/segger_dev/main)

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

- 📝 **[Sample Workflow](https://elihei2.github.io/segger_dev/notebooks/segger_tutorial/)**  
  Check out our tutorial showcasing a sample workflow with segger.

---

# Why segger?

- **Highly parallelizable** – Optimized for multi-GPU environments
- **Fast and efficient** – Trains in a fraction of the time compared to alternatives
- **Transfer learning** – Easily adaptable to new datasets and technologies

### Challenges in Segmentation

Spatial omics segmentation faces issues like:

- **Over/Under-segmentation**
- **Transcript contamination**
- **Scalability limitations**

segger tackles these with a **graph-based approach**, achieving superior segmentation accuracy.

---

## Installation Options

### Important: PyTorch Geometric Dependencies

Segger **relies heavily** on PyTorch Geometric for its graph-based operations. One **must** install its dependencies (such as `torch-sparse` and `torch-scatter`) based on their system’s specifications, especially the **CUDA** and **PyTorch** versions.

Please follow the official [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install the correct versions of `torch-sparse`, `torch-scatter`, and other relevant libraries.

Below is a quick guide for installing PyTorch Geometric dependencies for **torch 2.4.0**:

#### For CUDA 11.x:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

#### For CUDA 12.x:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

Afterwards choose the installation method that best suits your needs.

### Micromamba Installation

To set up Segger with `micromamba` and install the required dependencies, use the following commands:

```bash
micromamba create -n segger-rapids --channel-priority 1 \
    -c rapidsai -c conda-forge -c nvidia -c pytorch -c pyg \
    rapids=24.08 python=3.* 'cuda-version>=11.4,<=11.8' jupyterlab \
    'pytorch=*=*cuda*' 'pyg=*=*cu118' pyg-lib pytorch-sparse
micromamba install -n segger-rapids --channel-priority 1 --file mamba_environment.yml
micromamba run -n segger-rapids pip install --no-deps ./
```

### GitHub Installation

For a straightforward local installation from GitHub, clone the repository and install the package using `pip`:

```bash
git clone https://github.com/EliHei2/segger_dev.git
cd segger_dev
pip install -e "."
```

#### Pip Installation (RAPIDS and CUDA 11)

For installations requiring RAPIDS and CUDA 11 support, run:

```bash
pip install -e ".[rapids11]"
```

#### Pip Installation (RAPIDS and CUDA 12)

For installations requiring RAPIDS and CUDA 12 support, run:

```bash
pip install -e ".[rapids12]"
```

### Docker Installation

Segger provides an easy-to-use Docker container for those who prefer a containerized environment. To pull the latest Docker image:

```bash
docker pull danielunyi42/segger_dev:latest
```

The Docker image comes with all dependencies packaged, including RAPIDS. It currently supports only CUDA 12.2, and we will soon release a version that supports CUDA 11.8.

### Singularity Installation

For users who prefer Singularity, you can pull the Docker image as follows:

```bash
singularity pull docker://danielunyi42/segger_dev:latest
```

---

# Powered by

- **PyTorch Lightning & PyTorch Geometric**: Enables fast, efficient graph neural network (GNN) implementation for heterogeneous graphs.
- **Dask**: Scalable parallel processing and distributed task scheduling, ideal for handling large transcriptomic datasets.
- **Shapely & Geopandas**: Utilized for spatial operations such as polygon creation, scaling, and spatial relationship computations.
- **RAPIDS**: Provides GPU-accelerated computation for tasks like k-nearest neighbors (KNN) graph construction.
- **AnnData & Scanpy**: Efficient processing for single-cell datasets.
- **SciPy**: Facilitates spatial graph construction, including distance metrics and convex hull calculations for transcript clustering.

---

# Contributions

segger is **open-source** and welcomes contributions. Join us in advancing spatial omics segmentation!

- **Source Code**  
  [GitHub](https://github.com/EliHei2/segger_dev)

- **Bug Tracker**  
  [Report Issues](https://github.com/EliHei2/segger_dev/issues)

- **Full Documentation**  
  [API Reference](https://elihei2.github.io/segger_dev/api/)
