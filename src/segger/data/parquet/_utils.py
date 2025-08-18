import pandas as pd
import geopandas as gpd
import shapely
from shapely.affinity import scale
from pyarrow import parquet as pq
import numpy as np
import scipy as sp
from typing import Optional, List, Dict, Tuple, Set, Sequence
import sys
from types import SimpleNamespace
from pathlib import Path
import yaml
import os
import pyarrow as pa
import scanpy as sc
import anndata as ad
from itertools import combinations


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

    # Check if 'Geometry', 'geometry', 'polygon', or 'Polygon' is in the columns
    if any(col in columns for col in ['Geometry', 'geometry', 'polygon', 'Polygon']):
        import geopandas as gpd
        # If geometry columns are present, read with geopandas
        region = gpd.read_parquet(
            filepath,
            filters=filters,
            columns=columns,
        )
    else:
        # Otherwise, read with pandas
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
    scale_factor: float = 1.0,
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
    scale_factor : float, optional
        A ratio to scale the polygons. A value of 1.0 means no change,
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
        coords=boundaries[[x, y]].values.copy(order="C"),
        offsets=(geometry_offset, part_offset),
    )
    gs = gpd.GeoSeries(polygons, index=np.unique(ids))

    # print(gs)

    if scale_factor != 1.0:
        # Scale polygons around their centroid
        gs = gpd.GeoSeries(
            [
                scale(geom, xfact=scale_factor, yfact=scale_factor, origin='centroid')
                for geom in gs
            ],
            index=gs.index,
        )
        # print(gs)

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



def find_markers(
    adata: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 5,
    neg_percentile: float = 10,
    percentage: float = 50,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Identify positive and negative marker genes for each cell type in an AnnData object.
    
    Positive markers are top-ranked genes that are expressed in at least
    `percentage` percent of cells in the given cell type.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing gene expression data and cell type annotations.
    cell_type_column : str
        Name of the column in `adata.obs` specifying cell type identity for each cell.
    pos_percentile : float, optional (default: 5)
        Percentile threshold for selecting top highly expressed genes as positive markers.
    neg_percentile : float, optional (default: 10)
        Percentile threshold for selecting lowest expressed genes as negative markers.
    percentage : float, optional (default: 50)
        Minimum percent of cells (0-100) in a cell type expressing a gene for it to be a marker.
    
    Returns
    -------
    markers : dict
        Dictionary mapping cell type names to:
            {
                'positive': [list of positive marker gene names],
                'negative': [list of negative marker gene names]
            }
    """
    markers = {}
    sc.tl.rank_genes_groups(adata, groupby=cell_type_column)
    genes = np.array(adata.var_names)
    n_genes = adata.shape[1]

    # Work with a dense matrix for expression fraction calculation
    # (convert sparse to dense if needed)
    if not isinstance(adata.X, np.ndarray):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X

    for cell_type in adata.obs[cell_type_column].unique():
        mask = (adata.obs[cell_type_column] == cell_type).values
        gene_names = np.array(adata.uns['rank_genes_groups']['names'][cell_type])
        
        n_pos = max(1, int(n_genes * pos_percentile // 100))
        n_neg = max(1, int(n_genes * neg_percentile // 100))
        
        # Calculate percent of cells in this cell type expressing each gene
        expr_frac = (expr_matrix[mask] > 0).mean(axis=0) * 100  # as percent
        
        # Filter positive markers by expression fraction
        pos_indices = []
        for idx in range(n_pos):
            gene = gene_names[idx]
            gene_idx = np.where(genes == gene)[0][0]
            if expr_frac[gene_idx] >= percentage:
                pos_indices.append(idx)
        positive_markers = list(gene_names[pos_indices])
        
        # Negative markers are the lowest-ranked
        negative_markers = list(gene_names[-n_neg:])

        markers[cell_type] = {
            "positive": positive_markers,
            "negative": negative_markers
        }
    return markers

def find_mutually_exclusive_genes(
    adata: ad.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str
) -> List[Tuple[str, str]]:
    """Identify mutually exclusive genes based on expression criteria.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - markers: dict
        Dictionary where keys are cell types and values are dictionaries containing:
            'positive': list of top x% highly expressed genes
            'negative': list of top x% lowly expressed genes.
    - cell_type_column: str
        Column name in `adata.obs` that specifies cell types.

    Returns:
    - exclusive_pairs: list
        List of mutually exclusive gene pairs.
    """
    exclusive_genes = {}
    all_exclusive = []
    gene_expression = adata.to_df()
    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []
        for gene in positive_markers:
            gene_expr = adata[:, gene].X
            cell_type_mask = adata.obs[cell_type_column] == cell_type
            non_cell_type_mask = ~cell_type_mask
            if (gene_expr[cell_type_mask] > 0).mean() > 0.2 and (
                gene_expr[non_cell_type_mask] > 0
            ).mean() < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)
    unique_genes = list(
        {
            gene
            for i in exclusive_genes.keys()
            for gene in exclusive_genes[i]
            if gene in all_exclusive
        }
    )
    filtered_exclusive_genes = {
        i: [gene for gene in exclusive_genes[i] if gene in unique_genes]
        for i in exclusive_genes.keys()
    }
    mutually_exclusive_gene_pairs = [
        tuple(sorted((gene1, gene2)))
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        if key1 != key2
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]
    return set(mutually_exclusive_gene_pairs)



# def find_mutually_exclusive_genes(
#     adata: ad.AnnData, threshold: float = 0.0001
# ) -> List[Tuple[str, str]]:
#     """Identify pairs of genes with coexpression below a specified threshold.

#     Args:
#     - adata: AnnData
#         Annotated data object containing gene expression data.
#     - threshold: float
#         Coexpression threshold below which gene pairs are considered.

#     Returns:
#     - low_coexpression_pairs: list
#         List of gene pairs with low coexpression.
#     """
#     gene_expression = adata.to_df()
#     genes = gene_expression.columns
#     low_coexpression_pairs = []

#     for gene1, gene2 in combinations(genes, 2):
#         expr1 = gene_expression[gene1] > 0
#         expr2 = gene_expression[gene2] > 0
#         coexpression = (expr1 * expr2).mean()

#         if coexpression < threshold * (expr1.mean() + expr2.mean()):
#             low_coexpression_pairs.append(tuple(sorted((gene1, gene2))))

#     return set(low_coexpression_pairs)



# def find_mutually_exclusive_genes(
#     adata: ad.AnnData,
#     *,
#     threshold: float = 1e-4,
#     expr_cutoff: float = 0.0,
#     block_size: int = 2048,
# ) -> Set[Tuple[str, str]]:
#     """
#     Identify gene pairs (i, j) with coexpression below a specified threshold:
#         mean( expr_i & expr_j )  <  threshold * ( mean(expr_i) + mean(expr_j) )
#     computed via matrix operations on (cells x genes) data.

#     Parameters
#     ----------
#     adata : AnnData
#         Cells x Genes matrix (adata.X or adata.layers[layer]).
#     threshold : float, default 1e-4
#         Coexpression threshold weight for RHS.
#     layer : str or None, default None
#         Use adata.layers[layer] if provided; otherwise adata.X.
#     genes : sequence of str or None
#         Optional subset/order of genes to evaluate.
#     expr_cutoff : float, default 0.0
#         A cell expresses a gene if value > expr_cutoff.
#     block_size : int, default 2048
#         Number of genes per block for the blockwise sparse multiplication
#         to control memory (recommended: 1k–10k depending on RAM).

#     Returns
#     -------
#     low_coexp_pairs : set of (gene_i, gene_j)
#         Gene name pairs with i < j (lexicographic order preserved by indices).
#     """
#     # Select matrix and (optionally) subset genes
#     layer = None
#     genes = None 
#     X = adata.layers[layer] if layer is not None else adata.X
#     if genes is not None:
#         adata = adata[:, list(genes)]
#         X = adata.layers[layer] if layer is not None else adata.X

#     var_names = adata.var_names
#     n_cells, n_genes = adata.n_obs, adata.n_vars

#     # Binarize to a boolean CSR: expressed if > expr_cutoff
#     if sp.issparse(X):
#         Xb = X.tocsr().astype(np.float32)
#         Xb.data = (Xb.data > expr_cutoff).astype(np.uint8)
#         Xb.eliminate_zeros()
#     else:
#         Xb = sp.csr_matrix((X > expr_cutoff).astype(np.uint8))

#     # Per-gene expression fraction p(gene) = mean over cells
#     colsum = np.asarray(Xb.sum(axis=0)).ravel().astype(np.int64)  # counts of expressing cells
#     p = colsum / float(n_cells)  # shape (G,)

#     # We'll scan genes in blocks: for each block B, compute (Xb.T_B @ Xb) -> (B x G) intersection counts
#     result_pairs: Set[Tuple[str, str]] = set()
#     tN = threshold * n_cells  # scale to compare counts on LHS

#     for start in range(0, n_genes, block_size):
#         stop = min(start + block_size, n_genes)
#         # (cells x B)
#         Xb_block = Xb[:, start:stop]  # CSR
#         # Intersection counts for the block against all genes: (B x G)
#         inter_BG = (Xb_block.T @ Xb).tocoo()

#         # Build RHS for the whole block as a dense (B x G) using outer sums p_block[:,None] + p[None,:]
#         p_block = p[start:stop]  # (B,)
#         rhs_BG = tN * (p_block[:, None] + p[None, :])  # dense small block

#         # We need to test ALL pairs i in block, j in [0..G), not just where inter_BG has nonzeros.
#         # Strategy:
#         #   1) Start by assuming inter_ij = 0 for all pairs in the block (since absent in sparse).
#         #      For those, condition is: 0 < rhs_BG[i,j]  -> typically true unless rhs==0.
#         #   2) Then overwrite where we DO have nonzero intersections with the actual counts and re-test.
#         #
#         # Step 1: zero-intersection candidates (exclude diagonal and ensure i<j)
#         # We'll create a mask and then remove positions that are actually nonzero via a scatter step.

#         # Mask all positions initially; we'll clear diagonal and lower-triangle later
#         zero_mask = np.ones((stop - start, n_genes), dtype=bool)

#         # Clear the places where intersection is nonzero (we'll handle them separately)
#         zero_mask[inter_BG.row, inter_BG.col] = False

#         # Condition for zero-intersection pairs
#         # NOTE: We only add pairs where rhs > 0 (else the inequality cannot hold).
#         cand_mask = zero_mask & (rhs_BG > 0)

#         # Enforce i < j (global indices)
#         # Convert to global indices and filter upper triangle
#         if np.any(cand_mask):
#             rows, cols = np.where(cand_mask)
#             gi = rows + start
#             gj = cols
#             keep = gi < gj
#             gi, gj = gi[keep], gj[keep]
#             # Add pairs
#             for ii, jj in zip(gi, gj):
#                 result_pairs.add((var_names[ii], var_names[jj]))

#         # Step 2: handle nonzero intersections from inter_BG
#         if inter_BG.nnz:
#             gi = inter_BG.row + start
#             gj = np.asarray(inter_BG.col)
#             # Enforce i < j and exclude diagonal
#             keep = gi < gj
#             gi, gj = gi[keep], gj[keep]
#             inter_vals = inter_BG.data[keep].astype(np.float64)

#             # Compare: inter_ij  <  tN * (p_i + p_j)
#             rhs_vals = tN * (p[gi] + p[gj])
#             mask = inter_vals < rhs_vals

#             for ii, jj, ok in zip(gi, gj, mask):
#                 if ok:
#                     result_pairs.add((var_names[ii], var_names[jj]))

#         # help GC
#         del Xb_block, inter_BG, rhs_BG, zero_mask, cand_mask

#     return result_pairs