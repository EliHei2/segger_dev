Here's the updated `README.md` with a short introduction, focusing on the usage with clear and concise code examples:

---

# Segger

![Segger Model](docs/images/Segger_model_08_2024.png)

## Overview

Imaging-based spatial omics datasets present challenges in reliably segmenting single cells. Achieving accurate segmentation at single-cell resolution is crucial to unraveling multicellular mechanisms and understanding cell-cell communications in spatial omics studies. Despite the considerable progress and the variety of methods available for cell segmentation, challenges persist, including issues of over-segmentation, under-segmentation, and contamination from neighboring cells. 

Here we introduce Segger, a cell segmentation model designed for single-molecule resolved datasets, leveraging the co-occurrence of nucleic and cytoplasmic molecules (e.g., transcripts). It employs a heterogeneous graph structure on molecules and nuclei, integrating fixed-radius nearest neighbor graphs for nuclei and molecules, along with edges connecting transcripts to nuclei based on spatial proximity. A heterogeneous graph neural network (GNN) is then used to propagate information across these edges to learn the association of molecules with nuclei. Post-training, the model refines cell borders by regrouping transcripts based on confidence levels, overcoming issues like nucleus-less cells or overlapping cells.



## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/elihei2/segger_dev.git
cd segger_dev
pip install -r requirements.txt
```

## Usage

### Graph Dataset Building

Segger creates local heterogeneous graphs that cover both nuclei and nearby transcripts, capturing colocalization patterns and facilitating the study of spatial gene expression and nuclear architecture.

### Using Jupyter Notebook

Create a dataset with the provided Jupyter notebook located at `docs/notebooks/create_dataset.ipynb`:

```python
import os
import sys
from pathlib import Path
from src.segger.data.utils import XeniumSample

# Define paths
raw_data_dir = Path('data_raw/pancreatic')
processed_data_dir = Path('data_tidy/pyg_datasets/pancreatic')

# Load and process data
xs = XeniumSample().load_transcripts(path=raw_data_dir / 'transcripts.csv.gz', min_qv=30)
xs.load_nuclei(path=raw_data_dir / 'nucleus_boundaries.csv.gz')
xs.save_dataset_for_segger(
    processed_data_dir, 
    d_x=180, d_y=180, x_size=200, y_size=200, 
    r=3, val_prob=0.1, test_prob=0.1,
    k_nc=3, dist_nc=10, k_tx=5, dist_tx=3,
    compute_labels=True, sampling_rate=1
)
```

### Using Bash Script

Run the provided bash script to create the dataset:

```bash
./scripts/create_dataset.sh
```

### Using Python Script

Alternatively, run the Python script directly with command-line arguments:

```bash
python scripts/create_dataset.py \
    --raw_data_dir data_raw/pancreatic \
    --processed_data_dir data_tidy/pyg_datasets/pancreatic \
    --transcripts_url https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/transcripts.csv.gz \
    --nuclei_url https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/nucleus_boundaries.csv.gz \
    --min_qv 30 --d_x 180 --d_y 180 --x_size 200 --y_size 200 \
    --r 3 --val_prob 0.1 --test_prob 0.1 --k_nc 3 --dist_nc 10 \
    --k_tx 5 --dist_tx 3 --compute_labels True --sampling_rate 1
```

## Documentation

For detailed instructions, visit our [documentation](docs/index.html).

## License

Segger is licensed under the MIT License.

---

This version includes a brief introduction, installation instructions, and clear usage examples focusing on the code.
