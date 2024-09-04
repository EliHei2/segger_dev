import pandas as pd
import geopandas as gpd
import shapely
from pyarrow import parquet as pq
import numpy as np
import scipy as sp
from typing import Optional, List, Any, TYPE_CHECKING
import sys
import os
from itertools import cycle


if TYPE_CHECKING: # False at runtime
    import dask, cudf, dask_cudf


class TranscriptColumns():
    """
    A class to represent the column names for transcript data.

    Attributes
    ----------
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    id : str
        The name of the column representing the identifier.
    label : str
        The name of the column representing the feature name.
    xy : list of str
        A list containing the names of the x and y coordinate columns.
    """
    x = 'x_location'
    y = 'y_location'
    z = 'z_location'
    id = 'transcript_id'
    label = 'feature_name'
    nuclear = 'overlaps_nucleus'
    xy = [x, y]
    xyz = [x, y, z]
    columns = [x, y, z, label, nuclear, id, 'cell_id', 'qv']


class BoundaryColumns():
    """
    A class to represent the column names for boundary data.

    Attributes
    ----------
    x : str
        The name of the column representing the x-coordinate.
    y : str
        The name of the column representing the y-coordinate.
    id : str
        The name of the column representing the identifier.
    label : str
        The name of the column representing the cell identifier.
    xy : list of str
        A list containing the names of the x and y coordinate columns.
    """
    x = 'vertex_x'
    y = 'vertex_y'
    id = 'cell_id'
    label = 'cell_id'
    xy = [x, y]
    columns = [x, y, id]


# TODO: Add documentation
class BackendHandler:

    _valid_backends = {
        'pandas',
        'dask',
        'cudf',
        'dask_cudf',
    }

    def __init__(self, backend):
        # Make sure requested backend is supported
        if backend in self._valid_backends:
            self.backend = backend
        else:
            valid = ', '.join(map(lambda o: f"'{o}'", self._valid_backends))
            msg = f"Unsupported backend: {backend}. Valid options are {valid}."
            raise ValueError(msg)

        # Dynamically import packages only if requested
        if self.backend == 'pandas':
            import pandas as pd
        elif self.backend == 'dask':
            import dask
        elif self.backend == 'cudf':
            import cudf
        elif self.backend == 'dask_cudf':
            import dask_cudf
        else:
            raise ValueError('Internal Error')

    @property
    def read_parquet(self):
        if self.backend == 'pandas':
            return pd.read_parquet
        elif self.backend == 'dask':
            return dask.dataframe.read_parquet
        elif self.backend == 'cudf':
            return cudf.read_parquet
        elif self.backend == 'dask_cudf':
            return dask_cudf.read_parquet
        else:
            raise ValueError('Internal Error')


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
    backend: str = 'pandas',
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
    filters = [[
        (x, '>', bounds.bounds[0]),
        (y, '>', bounds.bounds[1]),
        (x, '<', bounds.bounds[2]),
        (y, '<', bounds.bounds[3]),
    ] + extra_filters]

    columns = list({x, y} | set(extra_columns))

    # Add split row groups options for dask backends
    kwargs = dict()
    if backend in ['dask', 'dask_cudf']:
        kwargs['split_row_groups'] = row_group_chunksize

    region = BackendHandler(backend).read_parquet(
        filepath,
        filters=filters,
        columns=columns,
        **kwargs,
    )
    return region


def get_polygons_from_xy(
    boundaries: pd.DataFrame,
) -> gpd.GeoSeries:
    """
    Convert boundary coordinates from a cuDF DataFrame to a GeoSeries of 
    polygons.

    Parameters
    ----------
    boundaries : pd.DataFrame
        A DataFrame containing the boundary data with x and y coordinates 
        and identifiers.

    Returns
    -------
    gpd.GeoSeries
        A GeoSeries containing the polygons created from the boundary 
        coordinates.
    """
    names = BoundaryColumns

    # Polygon offsets in coords
    ids = boundaries[names.label].values
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)

    # Convert to GeoSeries of polygons
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=boundaries[names.xy],
        offsets=(geometry_offset, part_offset),
    )
    gs = gpd.GeoSeries(polygons, index=np.unique(ids))

    return gs


def filter_boundaries(
    boundaries: pd.DataFrame,
    inset: shapely.Polygon,
    outset: shapely.Polygon,
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
    names = BoundaryColumns
    def in_region(region):
        in_x = boundaries[names.x].between(region.bounds[0], region.bounds[2])
        in_y = boundaries[names.y].between(region.bounds[1], region.bounds[3])
        return in_x & in_y
    x1, y1, x4, y4 = outset.bounds
    x2, y2, x3, y3 = inset.bounds
    boundaries['top'] = in_region(shapely.box(x1, y1, x4, y2))
    boundaries['left'] = in_region(shapely.box(x1, y1, x2, y4))
    boundaries['right'] = in_region(shapely.box(x3, y1, x4, y4))
    boundaries['bottom'] = in_region(shapely.box(x1, y3, x4, y4))
    boundaries['center'] = in_region(inset)

    # Filter boundary polygons
    # Include overlaps with top and left, not bottom and right
    gb = boundaries.groupby(names.id, sort=False)
    total = gb['center'].transform('size')
    in_top = gb['top'].transform('sum')
    in_left = gb['left'].transform('sum')
    in_right = gb['right'].transform('sum')
    in_bottom = gb['bottom'].transform('sum')
    in_center = gb['center'].transform('sum')
    keep = in_center == total
    keep |= ((in_center > 0) & (in_left > 0) & (in_bottom == 0))
    keep |= ((in_center > 0) & (in_top > 0) & (in_right == 0))
    inset_boundaries = boundaries.loc[keep]
    return inset_boundaries


def filter_transcripts(
    transcripts_df: pd.DataFrame,
    min_qv: float = 30.0,
) -> pd.DataFrame:
    """
    Filters transcripts based on quality value and removes unwanted transcripts.

    Parameters:
    transcripts_df (pd.DataFrame): The dataframe containing transcript data.
    min_qv (float): The minimum quality value threshold for filtering transcripts.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    filter_codewords = (
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "DeprecatedCodeword_",
    )
    mask = transcripts_df["qv"].ge(min_qv)
    mask &= ~transcripts_df["feature_name"].str.startswith(filter_codewords)
    return transcripts_df[mask]


# TODO: Add documentation
def estimate_density_from_metadata(
    filepath: os.PathLike,
    x: str,
    y: str,
    quantile: float,
):
    # Get index of columns of parquet file
    metadata = pq.read_metadata(filepath)
    x_idx = metadata.schema.names.index(x)
    y_idx = metadata.schema.names.index(y)

    # Calculate densities (nrows / area) of each row group from metadata
    densities = np.zeros(metadata.num_row_groups)
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        x_stats = group.column(x_idx).statistics
        y_stats = group.column(y_idx).statistics
        area = (x_stats.max - x_stats.min) * (y_stats.max - y_stats.min)
        densities[i] = group.num_rows /  area

    # Approximate distribution of densities in max row group
    dist = sp.stats.norm(densities.max(), scale=densities.std())
    return dist.ppf(quantile)


# TODO: Add documentation
def get_schema_size(filepath: os.PathLike, column: str):
    size_map = {
        'BOOLEAN': 1, 
        'INT32': 4,
        'FLOAT': 4,
        'INT64': 8,
        'DOUBLE': 8,
        'BYTE_ARRAY': 8,
        'INT96': 12,
    }
    metadata = pq.read_metadata(filepath)
    if column not in metadata.schema.names:
        raise KeyError(f"Column '{column}' not found in schema.")
    elif column == 'overlaps_nucleus':
        dtype = 'BOOLEAN'
    else:
        i = metadata.schema.names.index(column)
        dtype = metadata.schema[i].physical_type
    return size_map[dtype]


# TODO: Add documentation
def get_tile_size(filepath, x, y, tile):
    return read_parquet_region(filepath, x, y, tile).shape[0]


# TODO: Add documentation
# TODO: Check columns of gdf
def split_node(
    gdf: gpd.GeoDataFrame,
    idx: pd.Index,
    max_size: int,
    cycler: cycle = None,
) -> List:
    
    if cycler is None:
        cycler = cycle(('y', 'x'))

    # Edge case: empty index
    if len(idx) == 0:
        return []
    
    # Get cumulative sizes for each potential split point
    by = next(cycler)
    sums = gdf.loc[idx].groupby(by, sort=True)['n'].sum().cumsum()

    # Base case: singleton or below size requirement
    if len(idx) == 1 or sums.iloc[-1] <= max_size:
        return [idx]
    
    # Recursive case: gather leaves from left and right nodes
    split = sums.loc[sums <= sums.median()].idxmax()
    mask = gdf.loc[idx, by] <= split
    left_leaves = split_node(gdf, idx[mask], max_size, cycler)
    right_leaves = split_node(gdf, idx[~mask], max_size, cycler)

    # Combine and return the leaves
    return left_leaves + right_leaves

