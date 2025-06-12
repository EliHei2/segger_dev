import os
import cv2
import shapely
import tifffile
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Literal
from segger.io import _utils as utils
from skimage.transform import AffineTransform

EXTRACELLULAR_CODE = 0
NUCLEUS_CODE = 1
MEMBRANE_CODE = 2
CYTOPLASMIC_CODE = 3
MPP = 0.12028
TOL_FRAC = 1. / 50  # Fraction of area to simplify by

# NOTE: In CosMX, there is a bug in their segmentation where cell masks overlap
# with compartment masks from other cells (e.g. a cell mask A overlaps with 
# nuclear mask for cell B).

from pathlib import Path


def get_cosmx_polygons(
    cell_labels_dir: os.PathLike,
    compartment_labels_dir: os.PathLike,
    fov_positions_path: os.PathLike,
    compartment: Literal['cell', 'nucleus'],
) -> gpd.GeoDataFrame:
    """
    Extract cell or nuclear polygons from CosMX segmentation outputs.

    Parameters
    ----------
    cell_labels_dir : os.PathLike
        Directory containing CellLabels TIFFs (one per FOV).
    compartment_labels_dir : os.PathLike
        Directory containing CompartmentLabels TIFFs (one per FOV).
    fov_positions_file : os.PathLike
        CSV file with FOV positions and slide information.
    compartment : {'cell', 'nucleus'}
        Type of compartment to extract: 'cell' or 'nucleus'.

    Returns
    -------
    polygons : gpd.GeoDataFrame
        GeoDataFrame of polygons indexed by unique cell IDs, with FOV info.

    Raises
    ------
    ValueError
        If `compartment` is not valid or FOV file is missing columns.
    FileNotFoundError
        If expected TIFF files are missing.
    """
    # Explicitly coerce to Paths
    cell_labels_dir = Path(cell_labels_dir)
    compartment_labels_dir = Path(compartment_labels_dir)
    fov_positions_file = Path(fov_positions_path)

    # Check file and directory structures
    fov_info = pd.read_csv(fov_positions_path, index_col='FOV')
    _preflight_checks(cell_labels_dir, compartment_labels_dir, fov_info)

    # Add 'Slide' column if doesn't exist
    if 'Slide' not in fov_info:
        fov_info['Slide'] = 1

    # Check compartment type
    if compartment == 'cell':
        valid_codes = [NUCLEUS_CODE, MEMBRANE_CODE, CYTOPLASMIC_CODE]
    elif compartment == 'nucleus':
        valid_codes = [NUCLEUS_CODE]
    else:
        msg = f"Invalid compartment '{compartment}'. Choose 'cell' or 'nucleus'."
        raise ValueError(msg)

    # Assemble polygons per FOV
    polygons = []
    for fov, row in fov_info.iterrows():
        fov_id = str.zfill(str(fov), 3)
        cell_path = cell_labels_dir / f'CellLabels_F{fov_id}.tif'
        comp_path = compartment_labels_dir / f'CompartmentLabels_F{fov_id}.tif'

        # Get shapely polygons from cell masks
        cell_labels = tifffile.imread(cell_path)
        comp_labels = tifffile.imread(comp_path)
        masks = np.where(np.isin(comp_labels, valid_codes), cell_labels, 0)

        contours = utils.masks_to_contours(masks).swapaxes(0, -1)
        fov_poly = utils.contours_to_polygons(*contours)

        # Remove redundant vertices
        tol = np.sqrt(fov_poly.area).mean() * TOL_FRAC # scale by avg cell size
        fov_poly.geometry = fov_poly.geometry.simplify(tolerance=tol)

        # FOV coords -> Global coords
        tx = row['X_mm'] * 1e3 / MPP
        ty = row['Y_mm'] * 1e3 / MPP
        transform = AffineTransform(scale=[1, -1], translation=[-tx, -ty])
        fov_poly.geometry = shapely.transform(fov_poly.geometry, transform)

        prefix = f"c_{row['Slide']}_{fov}_"  # match CosMX ID structure
        fov_poly.index = prefix + fov_poly.index.astype(str)
        polygons.append(fov_poly)
    
    polygons = pd.concat(polygons)
    tx = fov_info['X_mm'].max() * 1e3 / MPP
    ty = fov_info['Y_mm'].max() * 1e3 / MPP
    transform = AffineTransform(translation=[tx, ty])
    polygons.geometry = shapely.transform(polygons.geometry, transform)
    
    return polygons #pd.concat(polygons)


def _preflight_checks(
    cell_dir: Path,
    comp_dir: Path,
    fov_info: pd.DataFrame,
) -> None:
    """
    Ensure input directories and FOV info file contain expected files and 
    columns.
    """
    required_cols = {'X_mm', 'Y_mm'}
    missing_cols = required_cols - set(fov_info.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in FOV info: {', '.join(missing_cols)}"
        )

    expected_fovs = [str.zfill(str(fov), 3) for fov in fov_info.index]
    expected_files = lambda prefix: {
        f"{prefix}_F{fov_id}.tif" for fov_id in expected_fovs
    }

    for directory, prefix in [
        (cell_dir, "CellLabels"),
        (comp_dir, "CompartmentLabels")
    ]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")
        actual = {f.name for f in directory.glob("*.tif")}
        expected = expected_files(prefix)
        missing = expected - actual
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} {prefix} TIFFs:\n" +
                "\n".join(sorted(missing))
            )
