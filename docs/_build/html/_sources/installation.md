
# Installation Guide

This guide provides detailed instructions for installing the `segger` package. Whether you are installing for CPU or GPU, the instructions below will guide you through the process.

```{warning}
Ensure you are using `Python >= 3.10` before starting the installation process.
```

## Install the Package


### From Source

To install `segger` from the source code:

1. Clone the repository:
    ```bash
    git clone https://github.com/EliHei2/segger_dev.git
    ```

2. Navigate to the project directory:
    ```bash
    cd segger_dev
    ```

3. Install the package:
    ```bash
    pip install .
    ```



## CPU and GPU Installation from PyPI

```{tab-set}

```{tab-item} CPU Installation
If you only need CPU support, use the following command:

```bash
pip install segger
```
This will install the package without any GPU-related dependencies.

```{note}
This is ideal for environments where GPU support is not required or available.
```


```{tab-item} GPU Installation
For installations with GPU support, use the following command:

```bash
pip install segger[gpu]
```

This includes the necessary dependencies for CUDA-enabled GPUs.

```{tip}
Ensure your machine has the appropriate CUDA drivers and NVIDIA libraries installed.
```

```

```

```



## Optional Dependencies

The following sections describe optional dependencies you can install for specific features.

```{tab-set}
```{tab-item} Torch Geometric

To install `torch-geometric` related dependencies, run:

```bash
pip install segger[torch-geometric]
```

Follow the additional steps on the [PyTorch Geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to ensure proper setup.

```{tip}
Ensure you install `torch-geometric` with the correct CUDA version for GPU support.
```

```{tab-item} Multiprocessing

To install `segger` with multiprocessing support, use the following command:

```bash
pip install segger[multiprocessing]
```

This will enable multi-core parallel processing features.

```

## Platform-Specific Installations

Below are instructions for installing `segger` on different operating systems.

```{tab-set}
```{tab-item} Linux

On Linux, use the following command to install the package:

```bash
pip install segger
```

For GPU support on Linux, ensure you have the necessary CUDA drivers installed:

```bash
pip install segger[gpu]
```



```{tab-item} macOS

To install on macOS, use the following command:

```bash
pip install segger
```

Note that macOS does not natively support CUDA, so GPU support is not available.

```{warning}
If you require GPU support, we recommend using Linux or Windows.
```

```{tab-item} Windows

To install on Windows, use the following command:

```bash
pip install segger
```

For GPU support on Windows:

```bash
pip install segger[gpu]
```

```{tip}
Ensure your CUDA drivers are installed on Windows by using `nvidia-smi`.
```

```

## Installation for Developers

For developers looking to contribute or work with `segger` in a development environment, you can install the package with development dependencies.

1. Clone the repository:
    ```bash
    git clone https://github.com/EliHei2/segger_dev.git
    ```

2. Navigate to the project directory:
    ```bash
    cd segger_dev
    ```

3. Install the package with development dependencies:
    ```bash
    pip install -e .[dev]
    ```

This will install additional dependencies such as `pytest`, `black`, `flake8`, and more.

## Common Installation Issues

```{tip}
If you encounter installation issues, ensure that you are using the correct Python version (`>= 3.10`) and that you have the necessary permissions to install packages on your system.
```

Some common errors include:

- **Missing Python version**: Ensure you are using `Python >= 3.10`.
- **Insufficient permissions**: Use `pip install --user` if you do not have admin permissions.
- **Conflicting CUDA drivers**: Ensure you have compatible CUDA versions installed if using GPU support.

For further troubleshooting, please refer to the [official documentation](https://github.com/EliHei2/segger_dev).


For more information, visit the [official GitHub repository](https://github.com/EliHei2/segger_dev).



```