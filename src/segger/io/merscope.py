import geopandas as gpd
import pandas as pd
import numpy as np
import shapely



def aggregate_merscope_polygons(
    boundaries: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate MERSCOPE polygon boundaries by fusing geometries associated with 
    the same EntityID across Z-indices.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        A GeoDataFrame containing polygon boundaries.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame where duplicated geometries are dissolved into a single
        fused polygon per EntityID, while singular entries are retained as-is.
    """

    # Heuristic to find fused geometries from vpt
    agg = boundaries.geometry.area.groupby(
        boundaries.EntityID
    ).nunique().ne(1)
    agg = boundaries.EntityID.map(agg)

    # De-duplicate singular or duplicated geometries
    singular = boundaries[~agg].groupby('EntityID').agg('first')

    # Union all fused geometries across z-stack
    fused = boundaries[agg].dissolve('EntityID', aggfunc={
        'ZIndex': lambda x: np.floor(np.mean(x)).astype(int),
        'ZLevel': 'mean',
        'ParentType': 'first',
        'ParentID': 'first',
        'Type': 'first',
        'Name': 'first',
    })
    boundaries = pd.concat([singular, fused])
    boundaries.index = boundaries.index.set_names(None).astype(str)

    # Aggregate any MultiPolygons
    agg = boundaries.geometry.apply(
        lambda x: type(x) == shapely.MultiPolygon
    )
    multi = boundaries.geometry.count_geometries().gt(1)

    # Get first Polygon in singular MultiPolygons
    mask = agg & ~multi
    boundaries.loc[mask, 'Geometry'] = boundaries[mask].geometry.apply(
        lambda g: g.geoms[0])
    
    # Otherwise get largest Polygon
    mask = agg & multi
    boundaries.loc[mask, 'Geometry'] = boundaries[mask].geometry.apply(
        lambda g: sorted(g.geoms, key=lambda g: g.area)[-1])

    return boundaries
