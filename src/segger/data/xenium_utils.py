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
    id = 'codeword_index'
    label = 'feature_name'
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
    id = 'label_id'
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