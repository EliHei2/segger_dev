# segger

## Installation

Install compatible versions of PyTorch and Lightning before installing **segger**.  
Optionally, install PyG-lib to accelerate heterogeneous graph operations.  
*Note: PyG-lib only supports CUDA ≤ 12.4 and PyTorch ≤ 2.5.0.*

- **PyTorch:** [Installation guide](https://pytorch.org/get-started/locally/)  
- **Lightning:** [Documentation](https://lightning.ai/docs/pytorch/stable/)  
- **PyG-lib:** [README](https://github.com/pyg-team/pyg-lib)

For example, on Linux with CUDA 12.1 and PyTorch 2.5.0:
```bash
# Install PyTorch and torchvision for CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
# Install Lightning (and TorchMetrics)
pip install lightning torchmetrics
# (Optional) Install PyG-lib
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

**March 2025:** Since **segger** is under active development, we recommend installing the latest version directly from GitHub:

```bash
git clone https://github.com/EliHei2/segger_dev.git segger
cd segger
pip install -e .
```