from numpy.typing import ArrayLike
from typing import List, Union
from pathlib import Path
import geopandas as gpd
import scanpy as sc
import pandas as pd
import scipy as sp
import numpy as np
import shapely
import skimage
import cv2

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


def masks_to_contours(masks: ArrayLike) -> np.ndarray:
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
    for p in props:
        # Get largest contour with label
        lbl_contours = cv2.findContours(
            np.pad(p.image, 0).astype('uint8'),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        contour = sorted(lbl_contours, key=lambda c: c.shape[0])[-1]
        if contour.shape[0] > 2:
            contour = np.hstack([
                np.squeeze(contour)[:, ::-1] + p.bbox[:2],  # vertices
                np.full((contour.shape[0], 1), p.label)  # ID
            ])
            contours.append(contour)
    contours = np.concatenate(contours)
    
    return contours


def contours_to_polygons(
    x: ArrayLike,
    y: ArrayLike,
    ids: ArrayLike,
) -> gpd.GeoDataFrame:
    """
    Convert contour vertices into Shapely polygons.

    Parameters
    ----------
    x : ArrayLike of shape (N,)
        x-coordinates of contour vertices.
    y : ArrayLike of shape (N,)
        y-coordinates of contour vertices.
    ids : ArrayLike of shape (N,)
        Cell ID for each (x, y) vertex. Contiguous vertices share the same ID.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing Shapely polygons, indexed by unique cell ID.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ids = np.asarray(ids)

    splits = np.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = np.hstack([0, splits, len(ids)])
    part_offset = np.arange(len(np.unique(ids)) + 1)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        coords=np.stack([x, y]).T.copy(order='C'),
        offsets=(geometry_offset, part_offset),
    )

    return gpd.GeoDataFrame(geometry=polygons, index=np.unique(ids))


def transcripts_to_anndata(
    transcripts: pd.DataFrame,
    cell_label: str,
    gene_label: str,
    coordinate_labels: List[str] = None,
):
    """
    Create an AnnData object from a transcript-level DataFrame.

    Parameters
    ----------
    transcripts : pd.DataFrame
        DataFrame containing transcript-level information with at least cell and
        gene labels.
    cell_label : str
        Column name in `transcripts` specifying cell identifiers.
    gene_label : str
        Column name in `transcripts` specifying gene identifiers.
    coordinate_labels : list of str, optional
        List of column names specifying spatial coordinates (e.g., ['x', 'y']).
        If provided, spatial centroids are computed and stored in 
        `obsm['X_spatial']`.

    Returns
    -------
    adata : sc.AnnData
        AnnData object with cells as observations and genes as variables. 
        Spatial coordinates are stored in `obsm['X_spatial']` if 
        `coordinate_labels` are provided.
    """
    # Feature names to indices
    ids_cell, labels_cell = pd.factorize(transcripts[cell_label])
    ids_gene, labels_gene = pd.factorize(transcripts[gene_label])
    
    # Remove NaN values
    mask = ids_cell >= 0
    ids_cell = ids_cell[mask]
    ids_gene = ids_gene[mask]
    
    # Sort row index
    order = np.argsort(ids_cell)
    ids_cell = ids_cell[order]
    ids_gene = ids_gene[order]
    
    # Build sparse matrix
    X = sp.sparse.coo_matrix(
        (
            np.ones_like(ids_cell),
            np.stack([ids_cell, ids_gene]),
        ),
        shape=(len(labels_cell), len(labels_gene)),
    ).tocsr()

    # To AnnData
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=labels_cell.astype(str)),
        var=pd.DataFrame(index=labels_gene),
    )
    adata.raw = adata.copy()

    # Add spatial coords
    if coordinate_labels is not None:
        coords = transcripts[coordinate_labels]
        centroids = coords.groupby(transcripts[cell_label]).mean()
        idx = adata.obs.index.astype(transcripts[cell_label].dtype)
        adata.obsm['X_spatial'] = centroids.loc[idx].values
    
    return adata
