## Preprocessing MERSCOPE Data with VPT for Segger

Segger requires nuclear masks as input, but these are often missing from raw MERSCOPE datasets. This guide shows how to generate them using [vizgen-postprocessing](https://github.com/Vizgen/vizgen-postprocessing), Vizgen's CLI tool for segmentation (installed as the `vpt` Python package).

### 1. Environment Setup

If you've already set up an environment for Segger, you likely have PyTorch installed. Otherwise, install it first (see [PyTorch install instructions](https://pytorch.org/get-started/locally/)). Then install `vpt` with Cellpose support:

```bash
python -m venv vpt-env
source vpt-env/bin/activate
pip install torch  # if not already installed
pip install "vpt[cellpose]"
```

### 2. Prepare Configuration

VPT uses a JSON file to define segmentation parameters like model type, channel index, and scaling. See [Vizgen's documentation](https://vizgen.github.io/vizgen-postprocessing/segmentation/configuration/) for details. An example config is available at `examples/vpt_nuclear_segmentation.json` in this repo.

### 3. Set Paths and Run VPT

```bash
export MERSCOPE_DATA_DIR=/path/to/merscope/data
export OUTPUTS_DIR=${MERSCOPE_DATA_DIR}/nuclear_segmentation

vpt run-segmentation \
  --segmentation-algorithm ${OUTPUTS_DIR}/vpt_nuclear_segmentation.json \
  --input-images ${MERSCOPE_DATA_DIR}/images/ \
  --input-micron-to-mosaic ${MERSCOPE_DATA_DIR}/images/micron_to_mosaic_pixel_transform.csv \
  --output-path ${OUTPUTS_DIR}/ \
  --tile-size 4800 \
  --tile-overlap 200
```

The resulting segmentation files can be used as input to Segger:

- transcripts: `${OUTPUTS_DIR}/detected_transcripts.csv`
- segmentation boundaries: `${OUTPUTS_DIR}/cellpose_nucleus_micron_space.parquet`
