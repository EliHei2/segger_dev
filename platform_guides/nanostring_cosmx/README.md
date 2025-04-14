## Extracting Nuclear Boundaries from CosMX Data for Segger

Segger uses nuclear boundaries as input, but CosMX datasets often only
provide labeled cell and compartment masks. This script extracts nuclear
boundaries from CosMX output and saves them in a format compatible with
the CosMX `*_polygons.csv` files.

### Expected Inputs

The script expects the following structure in the CosMX output directory:

```
CosMX_output_directory/
├── *_fov_positions_file.csv
├── CellLabels/
│   ├── CellLabels_F001.tif
│   ├── CellLabels_F002.tif
│   └── ...
└── CompartmentLabels/
    ├── CompartmentLabels_F001.tif
    ├── CompartmentLabels_F002.tif
    └── ...
```

These files are produced by NanoString’s CosMX pipeline. Nuclear masks are
extracted as the regions where the compartment label equals `1` (nucleus),
intersected with cell labels.

### Output

A single Parquet file named `<RUN>_nucleus_boundaries.parquet` will be
saved to the same directory. It matches the structure of CosMX's
`*_polygons.csv` file with the following columns:

- `cell`
- `fov`
- `cellID`
- `x_local_px`
- `y_local_px`
- `x_global_px`
- `y_global_px`

---

### 1. Environment Setup

Activate the environment where you have Segger and its dependencies
installed. If needed, install required packages like `numpy`, `pandas`,
`geopandas`, `tifffile`, `scikit-image`, `opencv-python`, and `shapely`.

---

### 2. Run Script

To extract nuclear boundaries, run:

```bash
python segment_nuclei.py /path/to/CosMX_output_directory
```

This will generate the output Parquet file in the same directory.

---

### Reference

For more information on CosMX data outputs, see  
[NanoString CosMX Output Documentation](https://nanostring.com/products/cosmx-spatial-molecular-imager/)
