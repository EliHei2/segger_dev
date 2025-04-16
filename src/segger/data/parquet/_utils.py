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
import os
import pyarrow as pa


def get_xy_extents(
    filepath,
    x: str,
    y: str,
) -> shapely.Polygon:
    """
    Get the bounding box of the x and y coordinates from a Parquet file.
    If the min/max statistics are not available, compute them efficiently from the data.

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
    # Process row groups
    metadata = pq.read_metadata(filepath)
    schema_idx = dict(map(reversed, enumerate(metadata.schema.names)))
    # Find min and max values across all row groups
    x_max = -1
    x_min = sys.maxsize
    y_max = -1
    y_min = sys.maxsize
    group = metadata.row_group(0)
    try:
        for i in range(metadata.num_row_groups):
            # print('*1')
            group = metadata.row_group(i)
            x_min = min(x_min, group.column(schema_idx[x]).statistics.min)
            x_max = max(x_max, group.column(schema_idx[x]).statistics.max)
            y_min = min(y_min, group.column(schema_idx[y]).statistics.min)
            y_max = max(y_max, group.column(schema_idx[y]).statistics.max)
            bounds = shapely.box(x_min, y_min, x_max, y_max)
    # If statistics are not available, compute them manually from the data
    except:
        import gc

        print(
            "metadata lacks the statistics of the tile's bounding box, computing might take longer!"
        )
        parquet_file = pd.read_parquet(filepath)
        x_col = parquet_file.loc[:, x]
        y_col = parquet_file.loc[:, y]
        del parquet_file
        gc.collect()
        x_min = x_col.min()
        y_min = y_col.min()
        x_max = x_col.max()
        y_max = y_col.max()
    bounds = shapely.geometry.box(x_min, y_min, x_max, y_max)
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
    buffer_ratio: float = 1.0,
) -> gpd.GeoSeries:
    """
    Convert boundary coordinates from a DataFrame to a GeoSeries of polygons.

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
    buffer_ratio : float, optional
        A ratio to expand or shrink the polygons. A value of 1.0 means no change,
        greater than 1.0 expands the polygons, and less than 1.0 shrinks the polygons
        (default is 1.0).

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

    if buffer_ratio != 1.0:
        # Calculate buffer distance based on polygon area
        areas = gs.area
        # Use the square root of the area to get a linear distance
        buffer_distances = np.sqrt(areas / np.pi) * (buffer_ratio - 1.0)
        # Apply buffer to each polygon with its specific distance
        gs = gpd.GeoSeries(
            [geom.buffer(dist) if dist != 0 else geom for geom, dist in zip(gs, buffer_distances)], index=gs.index
        )

    return gs


def compute_nuclear_transcripts(
    polygons: gpd.GeoSeries,
    transcripts: pd.DataFrame,
    x_col: str,
    y_col: str,
    nuclear_column: str = None,
    nuclear_value: str = None,
) -> pd.Series:
    """
    Computes which transcripts are nuclear based on their coordinates and the
    nuclear polygons.

    Parameters
    ----------
    polygons : gpd.GeoSeries
        The nuclear polygons
    transcripts : pd.DataFrame
        The transcripts DataFrame
    x_col : str
        The x-coordinate column name
    y_col : str
        The y-coordinate column name
    nuclear_column : str, optional
        The column name that indicates if a transcript is nuclear
    nuclear_value : str, optional
        The value in nuclear_column that indicates a nuclear transcript

    Returns
    -------
    pd.Series
        A boolean series indicating which transcripts are nuclear
    """
    # If nuclear_column and nuclear_value are provided, use them
    if nuclear_column is not None and nuclear_value is not None:
        if nuclear_column in transcripts.columns:
            return transcripts[nuclear_column].eq(nuclear_value)

    # Otherwise compute based on coordinates
    points = gpd.GeoSeries(gpd.points_from_xy(transcripts[x_col], transcripts[y_col]))
    return points.apply(lambda p: any(p.within(poly) for poly in polygons))


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
    qv_column: Optional[str] = None,
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
    qv_column : Optional[str]
        The name of the column representing the quality value.
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
    if min_qv is not None and qv_column is not None:
        mask &= transcripts_df[qv_column].ge(min_qv)
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
        msg = (
            f"Sample type '{sample_type}' not found in settings. "
            f"Available options: {', '.join(filenames)}"
        )
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


def add_transcript_ids(
    transcripts_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    id_col: str = "transcript_id",
    precision: int = 1000,
) -> pd.DataFrame:
    """
    Adds unique transcript IDs to a DataFrame based on x,y coordinates.

    Parameters
    ----------
    transcripts_df : pd.DataFrame
        DataFrame containing transcript data with x,y coordinates
    x_col : str
        Name of the x-coordinate column
    y_col : str
        Name of the y-coordinate column
    id_col : str, optional
        Name of the column to store the transcript IDs, default "transcript_id"
    precision : int, optional
        Precision multiplier for coordinate values to handle floating point precision,
        default 1000

    Returns
    -------
    pd.DataFrame
        DataFrame with added transcript_id column
    """
    # Create coordinate strings with specified precision
    x_coords = np.round(transcripts_df[x_col] * precision).astype(int).astype(str)
    y_coords = np.round(transcripts_df[y_col] * precision).astype(int).astype(str)
    coords_str = x_coords + "_" + y_coords

    # Generate unique IDs using a deterministic hash function
    def hash_coords(s):
        # Use a fixed seed for reproducibility
        seed = 1996
        # Combine string with seed and take modulo to get an 8-digit integer
        return abs(hash(s + str(seed))) % 100000000

    tx_ids = np.array([hash_coords(s) for s in coords_str], dtype=np.int32)

    # Add IDs to DataFrame
    transcripts_df = transcripts_df.copy()
    transcripts_df[id_col] = tx_ids

    return transcripts_df


def ensure_transcript_ids(
    parquet_path: os.PathLike,
    x_col: str,
    y_col: str,
    id_col: str = "transcript_id",
    precision: int = 1000,
) -> None:
    """
    Ensures that a parquet file has transcript IDs by adding them if missing.

    Parameters
    ----------
    parquet_path : os.PathLike
        Path to the parquet file
    x_col : str
        Name of the x-coordinate column
    y_col : str
        Name of the y-coordinate column
    id_col : str, optional
        Name of the column to store the transcript IDs, default "transcript_id"
    precision : int, optional
        Precision multiplier for coordinate values to handle floating point precision,
        default 1000
    """
    # First check metadata to see if column exists
    metadata = pq.read_metadata(parquet_path)
    schema_idx = dict(map(reversed, enumerate(metadata.schema.names)))

    # Only proceed if the column doesn't exist
    if id_col not in schema_idx:
        # Read the parquet file
        df = pd.read_parquet(parquet_path)

        # Add transcript IDs
        df = add_transcript_ids(df, x_col, y_col, id_col, precision)

        # Convert DataFrame to Arrow table
        table = pa.Table.from_pandas(df)

        # Write back to parquet
        pq.write_table(
            table,
            parquet_path,
            version="2.6",  # Use latest stable version
            write_statistics=True,  # Ensure statistics are written
            compression="snappy",  # Use snappy compression for better performance
        )
