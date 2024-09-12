import os
import sys
import gc
import shapely
import pyarrow
import numpy as np
from enum import Enum
import pandas as pd
import scipy as sp


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
    id = 'codeword_index'
    label = 'feature_name'
    xy = [x,y]


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
    id = 'label_id'
    label = 'cell_id'
    xy = [x,y]

pass

def get_xy_bounds(
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
    # Get index of x- and y-columns of parquet file
    metadata = pyarrow.parquet.read_metadata(filepath)
    for i in range(metadata.row_group(0).num_columns):
        column = metadata.row_group(0).column(i)
        if column.path_in_schema == x:
            x_idx = i
        elif column.path_in_schema == y:
            y_idx = i
    # Find min and max values across all row groups
    x_max = -1
    x_min = sys.maxsize
    y_max = -1
    y_min = sys.maxsize
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        x_min = min(x_min, group.column(x_idx).statistics.min)
        x_max = max(x_max, group.column(x_idx).statistics.max)
        y_min = min(y_min, group.column(y_idx).statistics.min)
        y_max = max(y_max, group.column(y_idx).statistics.max)
    bounds = shapely.box(x_min, y_min, x_max, y_max)
    return bounds


def read_parquet_region(
    filepath,
    x: str,
    y: str,
    bounds: shapely.Polygon = None,
    extra_columns: list[str] = [],
    extra_filters: list[str] = [],
    row_group_chunksize: int = 10,
) -> dask_cudf.DataFrame:
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
    dask_cudf.DataFrame
        A DataFrame containing the filtered data from the Parquet file.
    """
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

    columns = [x, y] + extra_columns
    region = dask_cudf.read_parquet(
        filepath,
        split_row_groups=row_group_chunksize,
        filters=filters,
        columns=columns,
    )
    return region


def get_polygons_from_xy(
    boundaries: cudf.DataFrame,
) -> cuspatial.GeoSeries:
    """
    Convert boundary coordinates from a cuDF DataFrame to a GeoSeries of 
    polygons.

    Parameters
    ----------
    boundaries : cudf.DataFrame
        A DataFrame containing the boundary data with x and y coordinates 
        and identifiers.

    Returns
    -------
    cuspatial.GeoSeries
        A GeoSeries containing the polygons created from the boundary 
        coordinates.
    """
    # Directly convert to GeoSeries from cuDF
    names = BoundaryColumns
    vertices = boundaries[names.xy].astype('double')
    ids = boundaries[names.id].values
    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = ring_offset = np.arange(len(np.unique(ids)) + 1)
    polygons = cuspatial.GeoSeries.from_polygons_xy(
        vertices.interleave_columns(),
        geometry_offset,
        part_offset,
        ring_offset,
    )
    del vertices
    gc.collect()
    return polygons


def get_points_from_xy(
    transcripts: cudf.DataFrame,
) -> cuspatial.GeoSeries:
    """
    Convert transcript coordinates from a cuDF DataFrame to a GeoSeries of 
    points.

    Parameters
    ----------
    transcripts : cudf.DataFrame
        A DataFrame containing the transcript data with x and y coordinates.

    Returns
    -------
    cuspatial.GeoSeries
        A GeoSeries containing the points created from the transcript 
        coordinates.
    """
    # Directly convert to GeoSeries from cuDF
    names = TranscriptColumns
    coords = transcripts[names.xy].astype('double')
    points = cuspatial.GeoSeries.from_points_xy(coords.interleave_columns())
    del coords
    gc.collect()
    return points


def filter_boundaries(
    boundaries: cudf.DataFrame,
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


def buffer_polygons(
    polygons: cuspatial.GeoSeries,
    distance: float,
):
    """
    Buffer (offset) polygons by a specified distance.

    Parameters
    ----------
    polygons : cuspatial.GeoSeries
        A GeoSeries containing the polygons to be buffered.
    distance : float
        The distance by which to buffer the polygons.

    Returns
    -------
    cuspatial.GeoSeries
        A GeoSeries containing the buffered polygons.
    """
    polygons_gp = polygons.to_geopandas()
    buffered = polygons_gp.buffer(distance)
    polygons_buffered = cuspatial.GeoSeries(buffered)
    del polygons_gp, buffered
    return polygons_buffered


def get_quadtree_kwargs(
    bounds: shapely.Polygon,
    user_quadtree_kwargs: dict = None,
):
    """
    Generate keyword arguments for creating a quadtree based on the given 
    bounds.

    Parameters
    ----------
    bounds : shapely.Polygon
        A polygon representing the bounding box for the quadtree.
    user_quadtree_kwargs : dict, optional
        A dictionary of user-specified keyword arguments to override the 
        default settings.

    Returns
    -------
    dict
        A dictionary containing the keyword arguments for creating a quadtree.

    Notes
    -----
    The function sets default values for `max_depth` and `max_size` if they 
    are not providedvin `user_quadtree_kwargs`. It also calculates the `scale` 
    parameter based on the size of the bounding box if it is not provided. The 
    final dictionary includes the minimum and maximum x and y coordinates of 
    the bounding box.
    """
    # Set default values for max_depth and max_size
    kwargs = {
        'max_depth': 10,
        'max_size': 10000
    }
    kwargs.update(user_quadtree_kwargs)

    # Calculate scale if not provided
    if 'scale' not in kwargs:
        x_range = bounds.bounds[2] - bounds.bounds[0]
        y_range = bounds.bounds[3] - bounds.bounds[1]
        kwargs['scale'] = (
            max(x_range, y_range) // (1 << kwargs['max_depth']) + 1
        )

    # Update kwargs with bounding box coordinates
    kwargs.update({
        'x_min': bounds.bounds[0],
        'y_min': bounds.bounds[1],
        'x_max': bounds.bounds[2],
        'y_max': bounds.bounds[3],
    })

    return kwargs


def get_expression_matrix(
    points: cuspatial.GeoSeries,
    points_idx: np.ndarray,
    polygons: cuspatial.GeoSeries,
    polygons_idx: np.ndarray,
    bounds: shapely.Polygon,
    quadtree_kwargs: dict = None,
):  
    """
    Generate a sparse expression matrix by assigning points to polygons based 
    on spatial relationships.

    Parameters
    ----------
    points : cuspatial.GeoSeries
        A GeoSeries containing the points (e.g., transcript locations).
    points_idx : np.ndarray
        An array of indices corresponding to the points, typically representing 
        gene identifiers.
    polygons : cuspatial.GeoSeries
        A GeoSeries containing the polygons (e.g., cell boundaries).
    polygons_idx : np.ndarray
        An array of indices corresponding to the polygons, typically 
        representing cell identifiers.
    bounds : shapely.Polygon
        A polygon representing the bounding box for the quadtree.
    quadtree_kwargs : dict, optional
        A dictionary of user-specified keyword arguments to override the 
        default settings for the quadtree.

    Returns
    -------
    sp.sparse.csr_array
        A sparse matrix where rows correspond to cells and columns correspond 
        to genes, with entries indicating the presence of a transcript in a 
        cell.

    Notes
    -----
    The function builds a quadtree on the points and uses it to efficiently 
    assign points to polygons based on their spatial relationships. The result 
    is a sparse matrix representing the expression levels of genes in cells.
    """
    # Keyword arguments reused below
    kwargs = get_quadtree_kwargs(bounds, quadtree_kwargs)
    
    # Build quadtree on points
    keys_to_pts, quadtree = cuspatial.quadtree_on_points(points, **kwargs)
    
    # Create bounding box and quadtree lookup
    kwargs.pop('max_size')  # not used below
    bboxes = cuspatial.polygon_bounding_boxes(polygons)
    poly_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree,
        bboxes,
        **kwargs
    )

    # Assign transcripts to cells based on polygon boundaries
    result = cuspatial.quadtree_point_in_polygon(
        poly_quad_pairs,
        quadtree,
        keys_to_pts,
        points,
        polygons,
    )
    
    # Map from transcript index to gene index
    codes = cudf.Series(points_idx)
    col_ind = result['point_index'].map(keys_to_pts).map(codes)
    col_ind = col_ind.to_numpy()
    
    # Get ordered cell IDs from Xenium
    _, row_uniques = pd.factorize(polygons_idx)
    row_ind = result['polygon_index'].map(cudf.Series(row_uniques))
    row_ind = row_ind.to_numpy() - 1  # originally, 1-index
    
    # Construct sparse expression matrix
    X = sp.sparse.csr_array(
        (np.ones(result.shape[0]), (row_ind, col_ind)),
        dtype=np.uint32,
    )
    return X


def get_buffered_counts(
    filepath_transcripts: os.PathLike,
    filepath_boundaries: os.PathLike,
    bounds: shapely.Polygon,
    buffer_distance: float,
    overlap: float = 100,
    quadtree_kwargs: dict = None,
):
    # Load transcripts
    outset = bounds.buffer(overlap, join_style='mitre')
    transcripts = read_parquet_region(
        filepath_transcripts,
        TranscriptColumns.x,
        TranscriptColumns.y,
        bounds=outset,
        extra_columns=[TranscriptColumns.id],
        extra_filters=[('qv', '>', 20)],
    ).compute()
    points = get_points_from_xy(transcripts)
    
    # Load boundaries
    boundaries = read_parquet_region(
        filepath_boundaries,
        BoundaryColumns.x,
        BoundaryColumns.y,
        bounds=outset,
        extra_columns=[BoundaryColumns.id]
    ).compute()
    boundaries = filter_boundaries(boundaries, bounds, outset)
    polygons = get_polygons_from_xy(boundaries)
    
    if buffer_distance != 0:
        polygons = buffer_polygons(polygons, buffer_distance)

    # Get sparse expression matrix
    X = get_expression_matrix(
        points,
        transcripts[TranscriptColumns.id].to_numpy(),
        polygons,
        boundaries[BoundaryColumns.id].to_numpy(),
        outset,
        quadtree_kwargs,
    )
    return X


def key_to_coordinate(key):
    """
    Convert a quadtree key to its corresponding x and y coordinates.

    Parameters
    ----------
    key : int
        The quadtree key to be converted.

    Returns
    -------
    pd.Series
        A pandas Series containing the y and x coordinates corresponding to 
        the quadtree key, with the index labels 'y' and 'x'.

    Notes
    -----
    The function converts the quadtree key to a binary string, splits it into 
    pairs of bits, and calculates the x and y coordinates by summing the 
    positions of the bits. The binary string is processed to ensure its length 
    is even by optionally prepending a '0'.
    """
    # Convert the key to binary and remove the '0b' prefix
    binary_key = bin(key)[2:]
    # Make sure the binary string length is even by optionally prepending a '0'
    if len(binary_key) % 2 != 0:
        binary_key = '0' + binary_key
    # Split the binary string into pairs
    pairs = [binary_key[i:i+2] for i in range(0, len(binary_key), 2)]
    # Initialize coordinates
    x, y = 0, 0
    # Iterate through each pair to calculate the sum of positions
    for i, pair in enumerate(pairs):
        power_of_2 = 2 ** (len(pairs) - i - 1)
        y += int(pair[0]) * power_of_2
        x += int(pair[1]) * power_of_2
    return pd.Series([y, x], index=['y', 'x'], name=key)


def get_quadrant_bounds(
    quadtree: pd.DataFrame,
    bounds: shapely.Polygon,
) -> pd.DataFrame:
    """
    Calculate the bounding box coordinates for each quadrant in a quadtree.

    Parameters
    ----------
    quadtree : pd.DataFrame
        A DataFrame representing the quadtree, with columns for 'key' and 
        'level'.
    bounds : shapely.Polygon
        A polygon representing the overall bounding box for the quadtree.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the original quadtree data and additional columns 
        'x_min', 'x_max', 'y_min', and 'y_max' representing the bounding box 
        coordinates for each quadrant.

    Notes
    -----
    The function calculates the bounding box coordinates for each quadrant 
    based on the quadtree key and level. It first converts the quadtree key to 
    x and y coordinates, then calculates the size of each quadrant at the 
    given level, and finally computes the minimum and maximum x and y 
    coordinates for each quadrant.
    """
    quadtree = quadtree.copy()
    x_min, y_min, x_max, y_max = bounds.bounds
    width = x_max - x_min
    height = y_max - y_min
    levels = quadtree['level'] + 1
    coords = quadtree['key'].apply(key_to_coordinate)
    quadrant_size_x = width / 2**levels 
    quadrant_size_y = height / 2**levels 
    
    quadtree['x_min'] = x_min + coords['x'] * quadrant_size_x
    quadtree['x_max'] = quadtree['x_min'] + quadrant_size_x
    quadtree['y_min'] = y_min + coords['y'] * quadrant_size_y
    quadtree['y_max'] = quadtree['y_min'] + quadrant_size_y
    return quadtree


def get_transcripts_regions(
    filepath,
    max_size: int = 1e7,
    bounds: shapely.Polygon = None,
):
    """
    Load transcript data and generate regions based on a quadtree structure.

    Parameters
    ----------
    filepath : str
        Path to the file containing transcript data.
    max_size : int, optional
        Maximum size for the quadtree nodes, by default 1e7.
    bounds : shapely.Polygon, optional
        A polygon representing the bounding box for the quadtree. If None, 
        bounds are calculated from the transcript data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the bounding box coordinates for each region 
        (quadrant) in the quadtree.
    """
    # Load transcripts
    if bounds is None:
        bounds = get_xy_bounds(filepath, *TranscriptColumns.xy)
    transcripts = read_parquet_region(
        filepath,
        TranscriptColumns.x,
        TranscriptColumns.y,
        bounds=bounds,
    ).compute()
    points = get_points_from_xy(transcripts)
    del transcripts
    gc.collect()
    
    # Build quadtree on points
    kwargs = dict(max_depth=10, max_size=max_size)
    kwargs = get_quadtree_kwargs(bounds, kwargs)
    _, quadtree = cuspatial.quadtree_on_points(points, **kwargs)
    quadtree_df = quadtree.to_pandas()
    del quadtree
    gc.collect()
    
    # Get boundaries of quadtree quadrants
    quadtree_df = get_quadrant_bounds(quadtree_df, bounds)
    regions = quadtree_df.loc[
        ~quadtree_df['is_internal_node'],
        ['x_min', 'y_min', 'x_max', 'y_max']
    ]
    return regions


def get_cell_labels(
    filepath_boundaries,
    row_group_chunksize: int = 10,
):
    """
    Load cell boundary data and generate cell labels.

    Parameters
    ----------
    filepath_boundaries : str
        Path to the file containing cell boundary data.
    row_group_chunksize : int, optional
        The number of row groups to read at a time, by default 10.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing cell labels.
    """
    id = BoundaryColumns.id
    label = BoundaryColumns.label
    boundaries = dask_cudf.read_parquet(
        filepath_boundaries,
        split_row_groups=row_group_chunksize,
        columns=[id, label],
    )
    boundaries[label] = boundaries[label].str.replace('\x00', '')
    cell_labels = boundaries[label].unique().compute()
    return cell_labels.to_numpy()


def get_buffered_counts_distributed(
    filepath_transcripts,
    filepath_boundaries,
    filepath_gene_panel,
    buffer_distance,
    client,
):
    """
    Generate buffered counts of transcripts within cell boundaries, 
    distributed across multiple workers.

    Parameters
    ----------
    filepath_transcripts : str
        Path to the file containing transcript data.
    filepath_boundaries : str
        Path to the file containing cell boundary data.
    filepath_gene_panel : str
        Path to the file containing gene panel data.
    buffer_distance : float
        Distance by which to buffer the cell boundaries.
    client : dask.distributed.Client
        Dask client for distributed computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the buffered counts of transcripts within cell 
        boundaries.

    Notes
    -----
    The function first splits the spatial region into quadrants and builds a 
    quadtree to distribute the workload evenly across multiple workers. It 
    then calculates the boundaries of each quadrant region and builds new 
    counts matrices for each region using buffered cell boundaries. Note that 
    transcripts can be doubly-counted if they fall within the buffered regions 
    of multiple cells.
    """
    # Get equal size regions of space w.r.t. no. transcripts in each region
    # This is a proxy for equally distributing workload
    # First split into quadrants to build quadtree on multiple workers; assumes
    # at least one split will be made in quadtree
    bounds = get_xy_bounds(filepath_transcripts, *TranscriptColumns.xy)
    x_min, y_min, x_max, y_max = bounds.buffer(1).bounds
    x_mid = (x_max - x_min) / 2 + x_min
    y_mid = (y_max - y_min) / 2 + y_min
    quadrants = [
        shapely.box(x_min, y_min, x_mid, y_mid),  # Q1
        shapely.box(x_min, y_mid, x_mid, y_max),  # Q2
        shapely.box(x_mid, y_min, x_max, y_mid),  # Q3
        shapely.box(x_mid, y_mid, x_max, y_max),  # Q4
    ]
    
    # Build quadtree and get boundaries of each quadrant region
    futures = client.map(
        lambda q: get_transcripts_regions(filepath_transcripts, bounds=q),
        quadrants
    )
    regions = pd.concat(client.gather(futures))
    gc.collect()
    
    # Build new counts matrices for each region using buffered (offset) cell
    # boundaries
    # Note: transcripts can be doubly-counted
    futures = client.map(
        lambda region: get_buffered_counts(
            filepath_transcripts,
            filepath_boundaries,
            shapely.box(*region),
            buffer_distance=buffer_distance,
        ),
        regions.values,
    )
    matrices = client.gather(futures)
    gc.collect()
