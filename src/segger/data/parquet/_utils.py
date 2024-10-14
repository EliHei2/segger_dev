import pandas as pd
import geopandas as gpd
import shapely
from pyarrow import parquet as pq
import numpy as np
import scipy as sp
from typing import Optional, List
import sys
from types import SimpleNamespace
from pathlib import Path
import yaml


def get_xy_extents(
    filepath,
    x: str,
    y: str,
) -> shapely.Polygon:
    """
    Get the bounding box of the x and y coordinates from a Parquet file.

    Parameters
    ----------
    filepath : str
        The path to the Parquet file.
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.

    Returns
    -------
    shapely.Polygon
        A polygon representing the bounding box of the x and y coordinates.
    """
    # Get index of columns of parquet file
    metadata = pq.read_metadata(filepath)
    schema_idx = dict(map(reversed, enumerate(metadata.schema.names)))

    # Find min and max values across all row groups
    x_max = -1
    x_min = sys.maxsize
    y_max = -1
    y_min = sys.maxsize
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        x_min = min(x_min, group.column(schema_idx[x]).statistics.min)
        x_max = max(x_max, group.column(schema_idx[x]).statistics.max)
        y_min = min(y_min, group.column(schema_idx[y]).statistics.min)
        y_max = max(y_max, group.column(schema_idx[y]).statistics.max)
    bounds = shapely.box(x_min, y_min, x_max, y_max)
    return bounds


def read_parquet_region(
    filepath,
    x: str,
    y: str,
    bounds: shapely.Polygon = None,
    extra_columns: list[str] = [],
    extra_filters: list[str] = [],
    row_group_chunksize: Optional[int] = None,
):
    """
    Read a region from a Parquet file based on x and y coordinates and optional
    filters.

    Parameters
    ----------
    filepath : str
        The path to the Parquet file.
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    bounds : shapely.Polygon, optional
        A polygon representing the bounding box to filter the data. If None,
        no bounding box filter is applied.
    extra_columns : list of str, optional
        A list of additional columns to include in the output DataFrame.
    extra_filters : list of str, optional
        A list of additional filters to apply to the data.

    Returns
    -------
    DataFrame
        A DataFrame containing the filtered data from the Parquet file.
    """
    # Check backend and load dependencies if not already loaded

    # Find bounds of full file if not supplied
    if bounds is None:
        bounds = get_xy_bounds(filepath, x, y)

    # Load pre-filtered data from Parquet file
    filters = [
        [
            (x, ">", bounds.bounds[0]),
            (y, ">", bounds.bounds[1]),
            (x, "<", bounds.bounds[2]),
            (y, "<", bounds.bounds[3]),
        ]
        + extra_filters
    ]

    columns = list({x, y} | set(extra_columns))

    region = pd.read_parquet(
        filepath,
        filters=filters,
        columns=columns,
    )
    return region


def get_polygons_from_xy(
    boundaries: pd.DataFrame,
    x: str,
    y: str,
    label: str,
) -> gpd.GeoSeries:
    """
    Convert boundary coordinates from a cuDF DataFrame to a GeoSeries of
    polygons.

    Parameters
    ----------
    boundaries : pd.DataFrame
        A DataFrame containing the boundary data with x and y coordinates
        and identifiers.
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    label : str
        The name of the column representing the cell or nucleus label.


    Returns
    -------
    gpd.GeoSeries
        A GeoSeries containing the polygons created from the boundary
        coordinates.
    """
    # Polygon offsets in coords
    ids = boundaries[label].values
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)

    # Convert to GeoSeries of polygons
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=boundaries[[x, y]],
        offsets=(geometry_offset, part_offset),
    )
    gs = gpd.GeoSeries(polygons, index=np.unique(ids))

    return gs


def filter_boundaries(
    boundaries: pd.DataFrame,
    inset: shapely.Polygon,
    outset: shapely.Polygon,
    x: str,
    y: str,
    label: str,
):
    """
    Filter boundary polygons based on their overlap with specified inset and
    outset regions.

    Parameters
    ----------
    boundaries : cudf.DataFrame
        A DataFrame containing the boundary data with x and y coordinates and
        identifiers.
    inset : shapely.Polygon
        A polygon representing the inner region to filter the boundaries.
    outset : shapely.Polygon
        A polygon representing the outer region to filter the boundaries.
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    label : str
        The name of the column representing the cell or nucleus label.

    Returns
    -------
    cudf.DataFrame
        A DataFrame containing the filtered boundary polygons.

    Notes
    -----
    The function determines overlaps of boundary polygons with the specified
    inset and outset regions. It creates boolean masks for overlaps with the
    top, left, right, and bottom sides of the outset region, as well as the
    center region defined by the inset polygon. The filtering logic includes
    polygons that:
    - Are completely within the center region.
    - Overlap with the center and the left side, but not the bottom side.
    - Overlap with the center and the top side, but not the right side.
    """

    # Determine overlaps of boundary polygons
    def in_region(region):
        in_x = boundaries[x].between(region.bounds[0], region.bounds[2])
        in_y = boundaries[y].between(region.bounds[1], region.bounds[3])
        return in_x & in_y

    x1, y1, x4, y4 = outset.bounds
    x2, y2, x3, y3 = inset.bounds
    boundaries["top"] = in_region(shapely.box(x1, y1, x4, y2))
    boundaries["left"] = in_region(shapely.box(x1, y1, x2, y4))
    boundaries["right"] = in_region(shapely.box(x3, y1, x4, y4))
    boundaries["bottom"] = in_region(shapely.box(x1, y3, x4, y4))
    boundaries["center"] = in_region(inset)

    # Filter boundary polygons
    # Include overlaps with top and left, not bottom and right
    gb = boundaries.groupby(label, sort=False)
    total = gb["center"].transform("size")
    in_top = gb["top"].transform("sum")
    in_left = gb["left"].transform("sum")
    in_right = gb["right"].transform("sum")
    in_bottom = gb["bottom"].transform("sum")
    in_center = gb["center"].transform("sum")
    keep = in_center == total
    keep |= (in_center > 0) & (in_left > 0) & (in_bottom == 0)
    keep |= (in_center > 0) & (in_top > 0) & (in_right == 0)
    inset_boundaries = boundaries.loc[keep]
    return inset_boundaries


def filter_transcripts(
    transcripts_df: pd.DataFrame,
    label: Optional[str] = None,
    filter_substrings: Optional[List[str]] = None,
    min_qv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filters transcripts based on quality value and removes unwanted transcripts.

    Parameters
    ----------
    transcripts_df : pd.DataFrame
        The dataframe containing transcript data.
    label : Optional[str]
        The label of transcript features.
    filter_substrings : Optional[str]
        The list of feature substrings to remove.
    min_qv : Optional[float]
        The minimum quality value threshold for filtering transcripts.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.
    """
    mask = pd.Series(True, index=transcripts_df.index)
    if filter_substrings is not None and label is not None:
        mask &= ~transcripts_df[label].str.startswith(tuple(filter_substrings))
    if min_qv is not None:
        mask &= transcripts_df["qv"].ge(min_qv)
    return transcripts_df[mask]


def load_settings(sample_type: str) -> SimpleNamespace:
    """
    Loads a matching YAML file from the _settings/ directory and converts its
    contents into a SimpleNamespace.

    Parameters
    ----------
    sample_type : str
        Name of the sample type to load (case-insensitive).

    Returns
    -------
    SimpleNamespace
        The settings loaded from the YAML file as a SimpleNamespace.

    Raises
    ------
    ValueError
        If `sample_type` does not match any filenames.
    """
    settings_dir = Path(__file__).parent.resolve() / "_settings"
    # Get a list of YAML filenames (without extensions) in the _settings dir
    filenames = [file.stem for file in settings_dir.glob("*.yaml")]
    # Convert sample_type to lowercase and check if it matches any filename
    sample_type = sample_type.lower()
    if sample_type not in filenames:
        msg = f"Sample type '{sample_type}' not found in settings. " f"Available options: {', '.join(filenames)}"
        raise FileNotFoundError(msg)
    # Load the matching YAML file
    yaml_file_path = settings_dir / f"{sample_type}.yaml"
    with yaml_file_path.open("r") as file:
        data = yaml.safe_load(file)

    # Convert the YAML data into a SimpleNamespace recursively
    return _dict_to_namespace(data)


def _dict_to_namespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace.
    """
    if isinstance(d, dict):
        d = {k: _dict_to_namespace(v) for k, v in d.items()}
        return SimpleNamespace(**d)
    return d
