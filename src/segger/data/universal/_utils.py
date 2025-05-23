from pyarrow import parquet as pq
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import torch
import json
import sys
import os

def get_xy_extents_parquet(
    filepath: os.PathLike,
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
    xmax = -1
    xmin = sys.maxsize
    ymax = -1
    ymin = sys.maxsize
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        xmin = min(xmin, group.column(schema_idx[x]).statistics.min)
        xmax = max(xmax, group.column(schema_idx[x]).statistics.max)
        ymin = min(ymin, group.column(schema_idx[y]).statistics.min)
        ymax = max(ymax, group.column(schema_idx[y]).statistics.max)
        bounds = shapely.box(xmin, ymin, xmax, ymax)
    return bounds

def get_xy_extents_geoparquet(filepath: os.PathLike) -> shapely.Polygon:
    """
    Extract the bounding box from the 'geo' metadata of a GeoParquet file.

    Parameters
    ----------
    filepath : os.PathLike
        Path to the GeoParquet file containing geometry metadata.

    Returns
    -------
    shapely.Polygon
        A polygon representing the spatial bounding box defined in the
        geometry metadata.
    """
    metadata = pq.read_schema(filepath).metadata
    metadata_geo = json.loads(metadata[b'geo'].decode('utf-8'))
    bounds = metadata_geo['columns']['geometry']['bbox']
    return shapely.box(*bounds)

   
def get_kdtree_edge_index(
    index_coords: np.ndarray,
    query_coords: np.ndarray,
    k: int,
    max_distance: float,
):
    """
    Computes the k-nearest neighbor edge indices using a KDTree.

    Parameters
    ----------
    index_coords : np.ndarray
        An array of shape (n_samples, n_features) representing the
        coordinates of the points to be indexed.
    query_coords : np.ndarray
        An array of shape (m_samples, n_features) representing the
        coordinates of the query points.
    k : int
        The number of nearest neighbors to find for each query point.
    max_distance : float
        The maximum distance to consider for neighbors.

    Returns
    -------
    torch.Tensor
        An array of shape (2, n_edges) containing the edge indices. Each
        column represents an edge between two points, where the first row
        contains the source indices and the second row contains the target
        indices.
    """
    # KDTree search
    tree = KDTree(index_coords)
    dist, idx = tree.query(query_coords, k, max_distance)

    # To sparse adjacency
    edge_index = np.argwhere(dist != np.inf).T
    edge_index[1] = idx[dist != np.inf]
    edge_index = torch.tensor(edge_index, dtype=torch.int64).contiguous()

    return edge_index

def get_polygon_props(
    polygons: gpd.GeoSeries,
    area: bool = True,
    convexity: bool = True,
    elongation: bool = True,
    circularity: bool = True,
) -> pd.DataFrame:
    """
    Computes geometric properties of polygons.

    Parameters
    ----------
    polygons : gpd.GeoSeries
        A GeoSeries containing polygon geometries.
    area : bool, optional
        If True, compute the area of each polygon (default is True).
    convexity : bool, optional
        If True, compute the convexity of each polygon (default is True).
    elongation : bool, optional
        If True, compute the elongation of each polygon (default is True).
    circularity : bool, optional
        If True, compute the circularity of each polygon (default is True).

    Returns
    -------
    props : pd.DataFrame
        A DataFrame containing the computed properties for each polygon.
    """
    props = pd.DataFrame(index=polygons.index, dtype=float)
    if area:
        props["area"] = polygons.area
    if convexity:
        props["convexity"] = polygons.convex_hull.area / polygons.area
    if elongation:
        rects = polygons.minimum_rotated_rectangle()
        props["elongation"] = rects.area / polygons.envelope.area
    if circularity:
        r = polygons.minimum_bounding_radius()
        props["circularity"] = polygons.area / r**2

    return props