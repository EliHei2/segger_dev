import os
import cv2
import shapely
import skimage
import tifffile
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt
from skimage.transform import ProjectiveTransform, AffineTransform

NUCLEUS_CODE = 1
MPP = 0.12028
TOL_FRAC = 1.0 / 50  # Fraction of area to simplify by

# NOTE: In CosMX, there is a bug in their segmentation where cell masks overlap
# with compartment masks from other cells (e.g. a cell mask A overlaps with
# nuclear mask for cell B).


def masks_to_contours(masks: np.ndarray) -> np.ndarray:
    """
    Convert labeled mask image to contours with cell ID annotations.

    Parameters
    ----------
    masks : np.ndarray
        A 2D array of labeled masks where each label corresponds to a cell.

    Returns
    -------
    np.ndarray
        An array of contour points with associated cell IDs.
    """
    # Get contour vertices from masks image
    props = skimage.measure.regionprops(masks.T)
    contours = []
    for i, p in enumerate(props):
        # Get largest contour with label
        lbl_contours = cv2.findContours(
            np.pad(p.image, 0).astype("uint8"),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        contour = sorted(lbl_contours, key=lambda c: c.shape[0])[-1]
        if contour.shape[0] > 2:
            contour = np.hstack(
                [
                    np.squeeze(contour)[:, ::-1] + p.bbox[:2],  # vertices
                    np.full((contour.shape[0], 1), i),  # ID
                ]
            )
            contours.append(contour)
    contours = np.concatenate(contours)

    return contours


def contours_to_polygons(
    contours: np.ndarray,
    transform: ProjectiveTransform = None,
) -> gpd.GeoSeries:
    """
    Convert contour vertices into Shapely polygons.

    Parameters
    ----------
    contours : np.ndarray
        Array of shape (3, N) where rows are x, y, and cell ID.
    transform : ProjectiveTransform, optional
        Transformation to apply to polygon coordinates.

    Returns
    -------
    gpd.GeoSeries
        A GeoSeries of polygons indexed by cell ID.
    """
    # Check shape of contours
    contours = np.array(contours)
    if contours.shape[0] != 3 and contours.shape[1] == 3:
        contours = contours.T
    # Convert to GeoSeries of Shapely polygons
    ids = contours[2]
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=contours[:2].T.copy(order="C"),
        offsets=(geometry_offset, part_offset),
    )
    if transform:
        polygons = shapely.transform(polygons, transform)
    return gpd.GeoSeries(polygons, index=np.unique(ids))


def get_file_match(file_dir: Union[str, Path], pattern: str) -> Path:
    """
    Find exactly one file in a directory matching the given pattern.

    Parameters
    ----------
    file_dir : Union[str, Path]
        Path to directory to search.
    pattern : str
        Glob pattern to match files.

    Returns
    -------
    Path
        Path to the matched file.

    Raises
    ------
    ValueError
        If no or multiple matches are found.
    """
    file_dir = Path(file_dir)
    matches = list(file_dir.glob(pattern))
    if len(matches) == 0:
        msg = f"No files found in {file_dir} matching pattern."
        raise ValueError(msg)
    if len(matches) > 1:
        msg = f"Multiple files found in {file_dir} matching pattern."
        raise ValueError(msg)
    return matches[0]


def write_nuclear_polygons(data_dir: os.PathLike):
    """
    Extract nuclear masks and write simplified polygons to Parquet.

    Parameters
    ----------
    data_dir : os.PathLike
        Root directory containing CosMX output including FOV CSV and
        label TIFFs. Output will be saved to this directory as well.
    """
    filepath = get_file_match(data_dir, "*_fov_positions_file.csv")
    fov_info = pd.read_csv(filepath, index_col="FOV")

    polygons = []
    for fov, row in fov_info.iterrows():
        # Convert mask array to polygons
        fov_id = str.zfill(str(fov), 3)
        cell_labels = tifffile.imread(
            data_dir / "CellLabels" / f"CellLabels_F{fov_id}.tif"
        )
        comp_labels = tifffile.imread(
            data_dir / "CompartmentLabels" / f"CompartmentLabels_F{fov_id}.tif"
        )
        masks = np.where(comp_labels == NUCLEUS_CODE, cell_labels, 0)
        contours = masks_to_contours(masks)
        fov_poly = contours_to_polygons(contours)

        # Simplify to reduce total number of vertices
        tol = np.sqrt(fov_poly.area).mean() * TOL_FRAC
        fov_poly = fov_poly.simplify(tolerance=tol)
        fov_poly = fov_poly.get_coordinates().reset_index()
        fov_poly.columns = ["cellID", "x_local_px", "y_local_px"]

        # Transform from local px to global px
        tx = row["X_mm"] * 1e3 / MPP
        ty = row["Y_mm"] * 1e3 / MPP
        tm = AffineTransform(
            scale=[1, -1],  # local coords are relative to the top of FOV
            translation=[tx, ty],  # move to global FOV position
        )
        fov_poly[["x_global_px", "y_global_px"]] = tm(
            fov_poly[["x_local_px", "y_local_px"]]
        )

        # Prepare data in format of CosMX polygons file
        fov_poly["fov"] = fov
        prefix = f"c_{row['Slide']}_{fov}_"
        fov_poly["cell"] = prefix + fov_poly["cellID"].astype(str)
        polygons.append(fov_poly)

    # Save to Parquet file
    polygons = pd.concat(polygons).sort_values(["fov", "cell"])
    run_name = fov_info["Run_Tissue_name"].unique()[0]
    polygons.to_parquet(
        data_dir / f"{run_name}_nucleus_boundaries.parquet",
        row_group_size=1000,
        engine="pyarrow",
        compression="snappy",
    )


def segment_cosmx_nuclei():
    """
    Command-line interface for writing nuclear polygons from CosMX output.

    Usage
    -----
    python script.py /path/to/data_dir
    """
    parser = argparse.ArgumentParser(
        description="Generate nucleus polygons from CosMX data."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to directory containing CosMX output files.",
    )
    args = parser.parse_args()
    write_nuclear_polygons(args.data_dir)


if __name__ == "__main__":
    segment_cosmx_nuclei()
