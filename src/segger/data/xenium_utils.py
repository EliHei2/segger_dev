import pandas as pd
import geopandas as gpd
import shapely
import numpy as np


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