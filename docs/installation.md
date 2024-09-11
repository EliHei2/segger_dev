

# Installation Guide

## Installation Options

Select the appropriate installation method based on your requirements.

=== "Micromamba Installation"
    ```bash
    micromamba create -n segger-rapids --channel-priority 1 \
        -c rapidsai -c conda-forge -c nvidia -c pytorch -c pyg \
        rapids=24.08 python=3.* 'cuda-version>=11.4,<=11.8' jupyterlab \
        'pytorch=*=*cuda*' 'pyg=*=*cu118' pyg-lib pytorch-sparse
    micromamba install -n segger-rapids --channel-priority 1 --file mamba_environment.yml
    micromamba run -n segger-rapids pip install --no-deps ./
    ```

=== "Github Installation"
    ```bash
    git clone https://github.com/EliHei2/segger_dev.git
    cd segger_dev
    pip install .
    ```




!!! warning "Common Installation Issues"
    - **Python Version**: Ensure you are using **Python >= 3.10**. Check your version with:
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
