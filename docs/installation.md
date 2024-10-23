## Segger Installation Guide

Select the appropriate installation method based on your requirements.

=== ":rocket: Micromamba Installation"
```bash
micromamba create -n segger-rapids --channel-priority 1 \
  -c rapidsai -c conda-forge -c nvidia -c pytorch -c pyg \
  rapids=24.10 python=3.* 'cuda-version>=12.0,<=12.1' jupyterlab \
  'pytorch=*=*cuda*' 'pyg=*=*cu121' pyg-lib pytorch-sparse
micromamba install -n segger-rapids --channel-priority 1 --file mamba_environment.yml
micromamba run -n segger-rapids pip install --no-deps ./
```

=== ":snake: Conda Installation"
```bash
conda create -n segger-env python=3.10
conda activate segger-env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
pip install .
```

=== ":whale: Docker Installation"
```bash
docker pull danielunyi42/segger_dev:cuda121
```

The Docker image comes with all required packages pre-installed, including PyTorch, RAPIDS, and PyTorch Geometric.
The current images support CUDA 11.8 and CUDA 12.1, which can be specified in the image tag.

For users who prefer Singularity:

```bash
singularity pull docker://danielunyi42/segger_dev:cuda121
```

=== ":octocat: Github Installation"
```bash
git clone https://github.com/EliHei2/segger_dev.git
cd segger_dev
pip install -e "."
```

=== ":rocket: Pip Installation of RAPIDS with CUDA 11"
```bash
pip install "segger[rapids11]"
```

=== ":rocket: Pip Installation of RAPIDS with CUDA 12"
```bash
pip install "segger[rapids12]"
```

!!! warning "Common Installation Issues"
    - **Python Version**: Ensure you are using Python >= 3.10. Check your version with:
      ```bash
      python --version
      ```
      If necessary, upgrade to the correct version.

    - **CUDA Compatibility (GPU)**: For GPU installations, ensure the correct CUDA drivers are installed. Verify your setup with:
      ```bash
      nvidia-smi
      ```
      Ensure your CUDA version is compatible with the package.

    - **Permissions**: If you encounter permission errors, use the `--user` flag to install without admin rights:
      ```bash
      pip install --user .
      ```
