# Function to attempt importing a module and warn if it's not installed
def try_import(module_name):
    try:
        globals()[module_name] = __import__(module_name)
    except ImportError:
        print(f"Warning: {module_name} is not installed. Please install it to use this functionality.")


# Standard imports
import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, List, Callable, Tuple
from shapely.geometry import Polygon
import geopandas as gpd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData, InMemoryDataset, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import radius_graph
import os
from scipy.spatial import cKDTree

# import hnswlib
from shapely.geometry import Polygon
from shapely.affinity import scale
import dask.dataframe as dd
from pyarrow import parquet as pq
import sys

# Attempt to import specific modules with try_import function
try_import("multiprocessing")
try_import("joblib")
# try_import("cuvs")
# try:
#     import cupy as cp
#     from cuvs.neighbors import cagra
# except ImportError:
#     print(f"Warning: cupy and/or cuvs are not installed. Please install them to use this functionality.")

import torch.utils.dlpack as dlpack
from datetime import timedelta


def filter_transcripts( #ONLY FOR XENIUM
    transcripts_df: pd.DataFrame,
    min_qv: float = 20.0,
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
        "UnassignedCodeword_",
    )
    
    transcripts_df['feature_name'] = transcripts_df['feature_name'].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )
    mask_quality = transcripts_df['qv'] >= min_qv

    # Apply the filter for unwanted codewords using Dask string functions
    mask_codewords = ~transcripts_df['feature_name'].str.startswith(filter_codewords)

    # Combine the filters and return the filtered Dask DataFrame
    mask = mask_quality & mask_codewords
    return transcripts_df[mask]


def compute_transcript_metrics(
    df: pd.DataFrame, qv_threshold: float = 30, cell_id_col: str = "cell_id"
) -> Dict[str, Any]:
    """
    Computes various metrics for a given dataframe of transcript data filtered by quality value threshold.

    Parameters:
        df (pd.DataFrame): The dataframe containing transcript data.
        qv_threshold (float): The quality value threshold for filtering transcripts.
        cell_id_col (str): The name of the column representing the cell ID.

    Returns:
        Dict[str, Any]: A dictionary containing various transcript metrics:
            - 'percent_assigned' (float): The percentage of assigned transcripts.
            - 'percent_cytoplasmic' (float): The percentage of cytoplasmic transcripts among assigned transcripts.
            - 'percent_nucleus' (float): The percentage of nucleus transcripts among assigned transcripts.
            - 'percent_non_assigned_cytoplasmic' (float): The percentage of non-assigned cytoplasmic transcripts.
            - 'gene_metrics' (pd.DataFrame): A dataframe containing gene-level metrics.
    """
    df_filtered = df[df["qv"] > qv_threshold]
    total_transcripts = len(df_filtered)
    assigned_transcripts = df_filtered[df_filtered[cell_id_col] != -1]
    percent_assigned = len(assigned_transcripts) / (total_transcripts + 1) * 100
    cytoplasmic_transcripts = assigned_transcripts[assigned_transcripts["overlaps_nucleus"] != 1]
    percent_cytoplasmic = len(cytoplasmic_transcripts) / (len(assigned_transcripts) + 1) * 100
    percent_nucleus = 100 - percent_cytoplasmic
    non_assigned_transcripts = df_filtered[df_filtered[cell_id_col] == -1]
    non_assigned_cytoplasmic = non_assigned_transcripts[non_assigned_transcripts["overlaps_nucleus"] != 1]
    percent_non_assigned_cytoplasmic = len(non_assigned_cytoplasmic) / (len(non_assigned_transcripts) + 1) * 100
    gene_group_assigned = assigned_transcripts.groupby("feature_name")
    gene_group_all = df_filtered.groupby("feature_name")
    gene_percent_assigned = (gene_group_assigned.size() / (gene_group_all.size() + 1) * 100).reset_index(
        names="percent_assigned"
    )
    cytoplasmic_gene_group = cytoplasmic_transcripts.groupby("feature_name")
    gene_percent_cytoplasmic = (cytoplasmic_gene_group.size() / (len(cytoplasmic_transcripts) + 1) * 100).reset_index(
        name="percent_cytoplasmic"
    )
    gene_metrics = pd.merge(gene_percent_assigned, gene_percent_cytoplasmic, on="feature_name", how="outer").fillna(0)
    results = {
        "percent_assigned": percent_assigned,
        "percent_cytoplasmic": percent_cytoplasmic,
        "percent_nucleus": percent_nucleus,
        "percent_non_assigned_cytoplasmic": percent_non_assigned_cytoplasmic,
        "gene_metrics": gene_metrics,
    }
    return results


def create_anndata(
    df: pd.DataFrame,
    panel_df: Optional[pd.DataFrame] = None,
    min_transcripts: int = 5,
    cell_id_col: str = "cell_id",
    qv_threshold: float = 30,
    min_cell_area: float = 10.0,
    max_cell_area: float = 1000.0,
) -> ad.AnnData:
    """
    Generates an AnnData object from a dataframe of segmented transcriptomics data.

    Parameters:
        df (pd.DataFrame): The dataframe containing segmented transcriptomics data.
        panel_df (Optional[pd.DataFrame]): The dataframe containing panel information.
        min_transcripts (int): The minimum number of transcripts required for a cell to be included.
        cell_id_col (str): The column name representing the cell ID in the input dataframe.
        qv_threshold (float): The quality value threshold for filtering transcripts.
        min_cell_area (float): The minimum cell area to include a cell.
        max_cell_area (float): The maximum cell area to include a cell.

    Returns:
        ad.AnnData: The generated AnnData object containing the transcriptomics data and metadata.
    """
    # Filter out unassigned cells
    df_filtered = df[df[cell_id_col].astype(str) != "UNASSIGNED"]

    # Create pivot table for gene expression counts per cell
    pivot_df = df_filtered.rename(columns={cell_id_col: "cell", "feature_name": "gene"})[["cell", "gene"]].pivot_table(
        index="cell", columns="gene", aggfunc="size", fill_value=0
    )
    pivot_df = pivot_df[pivot_df.sum(axis=1) >= min_transcripts]

    # Summarize cell metrics
    cell_summary = []
    for cell_id, cell_data in df_filtered.groupby(cell_id_col):
        if len(cell_data) < min_transcripts:
            continue
        cell_convex_hull = ConvexHull(cell_data[["x_location", "y_location"]], qhull_options="QJ")
        cell_area = cell_convex_hull.area
        if cell_area < min_cell_area or cell_area > max_cell_area:
            continue
        cell_summary.append(
            {
                "cell": cell_id,
                "cell_centroid_x": cell_data["x_location"].mean(),
                "cell_centroid_y": cell_data["y_location"].mean(),
                "cell_area": cell_area,
            }
        )
    cell_summary = pd.DataFrame(cell_summary).set_index("cell")

    # Add genes from panel_df (if provided) to the pivot table
    if panel_df is not None:
        panel_df = panel_df.sort_values("gene")
        genes = panel_df["gene"].values
        for gene in genes:
            if gene not in pivot_df:
                pivot_df[gene] = 0
        pivot_df = pivot_df[genes.tolist()]

    # Create var DataFrame
    if panel_df is None:
        var_df = pd.DataFrame(
            [
                {"gene": gene, "feature_types": "Gene Expression", "genome": "Unknown"}
                for gene in np.unique(pivot_df.columns.values)
            ]
        ).set_index("gene")
    else:
        var_df = panel_df[["gene", "ensembl"]].rename(columns={"ensembl": "gene_ids"})
        var_df["feature_types"] = "Gene Expression"
        var_df["genome"] = "Unknown"
        var_df = var_df.set_index("gene")

    # Compute total assigned and unassigned transcript counts for each gene
    assigned_counts = df_filtered.groupby("feature_name")["feature_name"].count()
    unassigned_counts = df[df[cell_id_col].astype(str) == "UNASSIGNED"].groupby("feature_name")["feature_name"].count()
    var_df["total_assigned"] = var_df.index.map(assigned_counts).fillna(0).astype(int)
    var_df["total_unassigned"] = var_df.index.map(unassigned_counts).fillna(0).astype(int)

    # Filter cells and create the AnnData object
    cells = list(set(pivot_df.index) & set(cell_summary.index))
    pivot_df = pivot_df.loc[cells, :]
    cell_summary = cell_summary.loc[cells, :]
    adata = ad.AnnData(pivot_df.values)
    adata.var = var_df
    adata.obs["transcripts"] = pivot_df.sum(axis=1).values
    adata.obs["unique_transcripts"] = (pivot_df > 0).sum(axis=1).values
    adata.obs_names = pivot_df.index.values.tolist()
    adata.obs = pd.merge(adata.obs, cell_summary.loc[adata.obs_names, :], left_index=True, right_index=True)

    return adata


def calculate_gene_celltype_abundance_embedding(adata: ad.AnnData, celltype_column: str) -> pd.DataFrame:
    """Calculate the cell type abundance embedding for each gene based on the fraction of cells in each cell type
    that express the gene (non-zero expression).

    Parameters:
        adata (ad.AnnData): An AnnData object containing gene expression data and cell type information.
        celltype_column (str): The column name in `adata.obs` that contains the cell type information.

    Returns:
        pd.DataFrame: A DataFrame where rows are genes and columns are cell types, with each value representing
            the fraction of cells in that cell type expressing the gene.

    Example:
        >>> adata = AnnData(...)  # Load your scRNA-seq AnnData object
        >>> celltype_column = 'celltype_major'
        >>> abundance_df = calculate_gene_celltype_abundance_embedding(adata, celltype_column)
        >>> abundance_df.head()
    """
    # Extract expression data (cells x genes) and cell type information (cells)
    expression_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    cell_types = adata.obs[celltype_column].values
    # Create a binary matrix for gene expression (1 if non-zero, 0 otherwise)
    gene_expression_binary = (expression_data > 0).astype(int)
    # Convert the binary matrix to a DataFrame
    gene_expression_df = pd.DataFrame(gene_expression_binary, index=adata.obs_names, columns=adata.var_names)
    # Perform one-hot encoding on the cell types
    encoder = OneHotEncoder(sparse_output=False)
    cell_type_encoded = encoder.fit_transform(cell_types.reshape(-1, 1))
    # Calculate the fraction of cells expressing each gene per cell type
    cell_type_abundance_list = []
    for i in range(cell_type_encoded.shape[1]):
        # Extract cells of the current cell type
        cell_type_mask = cell_type_encoded[:, i] == 1
        # Calculate the abundance: sum of non-zero expressions in this cell type / total cells in this cell type
        abundance = gene_expression_df[cell_type_mask].mean(axis=0)
        cell_type_abundance_list.append(abundance)
    # Create a DataFrame for the cell type abundance with gene names as rows and cell types as columns
    cell_type_abundance_df = pd.DataFrame(
        cell_type_abundance_list, columns=adata.var_names, index=encoder.categories_[0]
    ).T
    return cell_type_abundance_df


def get_edge_index(
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    k: int = 5,
    dist: int = 10,
    method: str = "kd_tree",
    workers: int = 1,
) -> torch.Tensor:
    """
    Computes edge indices using KD-Tree.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.
        method (str, optional): The method to use. Only 'kd_tree' is supported now.

    Returns:
        torch.Tensor: Edge indices.
    """
    if method == "kd_tree":
        return get_edge_index_kdtree(coords_1, coords_2, k=k, dist=dist, workers=workers)
    # elif method == "cuda":
    #     return get_edge_index_cuda(coords_1, coords_2, k=k, dist=dist)
    else:
        msg = f"Unknown method {method}. The only supported method is 'kd_tree' now."
        raise ValueError(msg)


def get_edge_index_kdtree(
    coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10, workers: int = 1
) -> torch.Tensor:
    """
    Computes edge indices using KDTree.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.

    Returns:
        torch.Tensor: Edge indices.
    """
    if isinstance(coords_1, torch.Tensor):
        coords_1 = coords_1.cpu().numpy()
    if isinstance(coords_2, torch.Tensor):
        coords_2 = coords_2.cpu().numpy()
    tree = cKDTree(coords_1)
    d_kdtree, idx_out = tree.query(coords_2, k=k, distance_upper_bound=dist, workers=workers)
    valid_mask = d_kdtree < dist
    edges = []

    for idx, valid in enumerate(valid_mask):
        valid_indices = idx_out[idx][valid]
        if valid_indices.size > 0:
            edges.append(np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T)

    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index


# def get_edge_index_cuda(
#     coords_1: torch.Tensor,
#     coords_2: torch.Tensor,
#     k: int = 10,
#     dist: float = 10.0,
#     metric: str = "sqeuclidean",
#     nn_descent_niter: int = 100,
# ) -> torch.Tensor:
#     """
#     Computes edge indices using RAPIDS cuVS with cagra for vector similarity search,
#     with input coordinates as PyTorch tensors on CUDA, using DLPack for conversion.

#     Parameters:
#         coords_1 (torch.Tensor): First set of coordinates (query vectors) on CUDA.
#         coords_2 (torch.Tensor): Second set of coordinates (index vectors) on CUDA.
#         k (int, optional): Number of nearest neighbors.
#         dist (float, optional): Distance threshold.

#     Returns:
#         torch.Tensor: Edge indices as a PyTorch tensor on CUDA.
#     """

#     def cupy_to_torch(cupy_array):
#         return torch.from_dlpack((cupy_array.toDlpack()))

#     # gg
#     def torch_to_cupy(tensor):
#         return cp.fromDlpack(dlpack.to_dlpack(tensor))

#     # Convert PyTorch tensors (CUDA) to CuPy arrays using DLPack
#     cp_coords_1 = torch_to_cupy(coords_1).astype(cp.float32)
#     cp_coords_2 = torch_to_cupy(coords_2).astype(cp.float32)
#     # Define the distance threshold in CuPy
#     cp_dist = cp.float32(dist)
#     # IndexParams and SearchParams for cagra
#     # compression_params = cagra.CompressionParams(pq_bits=pq_bits)
#     index_params = cagra.IndexParams(
#         metric=metric, nn_descent_niter=nn_descent_niter
#     )  # , compression=compression_params)
#     search_params = cagra.SearchParams()
#     # Build index using CuPy coords
#     try:
#         index = cagra.build(index_params, cp_coords_1)
#     except AttributeError:
#         index = cagra.build_index(index_params, cp_coords_1)
#     # Perform search to get distances and indices (still in CuPy)
#     D, I = cagra.search(search_params, index, cp_coords_2, k)
#     # Boolean mask for filtering distances below the squared threshold (all in CuPy)
#     valid_mask = cp.asarray(D < cp_dist**2)
#     # Vectorized operations for row and valid indices (all in CuPy)
#     repeats = valid_mask.sum(axis=1).tolist()
#     row_indices = cp.repeat(cp.arange(len(cp_coords_2)), repeats)
#     valid_indices = cp.asarray(I)[cp.where(valid_mask)]
#     # Stack row indices with valid indices to form edges
#     edges = cp.vstack((row_indices, valid_indices)).T
#     # Convert the result back to a PyTorch tensor using DLPack
#     edge_index = cupy_to_torch(edges).long().contiguous()
#     return edge_index


class SpatialTranscriptomicsDataset(InMemoryDataset):
    """A dataset class for handling SpatialTranscriptomics spatial transcriptomics data.

    Attributes:
        root (str): The root directory where the dataset is stored.
        transform (callable): A function/transform that takes in a Data object and returns a transformed version.
        pre_transform (callable): A function/transform that takes in a Data object and returns a transformed version.
        pre_filter (callable): A function that takes in a Data object and returns a boolean indicating whether to keep it.
    """

    def __init__(
        self, root: str, transform: Callable = None, pre_transform: Callable = None, pre_filter: Callable = None
    ):
        """Initialize the SpatialTranscriptomicsDataset.

        Args:
            root (str): Root directory where the dataset is stored.
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version. Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version. Defaults to None.
            pre_filter (callable, optional): A function that takes in a Data object and returns a boolean indicating whether to keep it. Defaults to None.
        """
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """Return a list of raw file names in the raw directory.

        Returns:
            List[str]: List of raw file names.
        """
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self) -> List[str]:
        """Return a list of processed file names in the processed directory.

        Returns:
            List[str]: List of processed file names.
        """
        return [x for x in os.listdir(self.processed_dir) if "tiles" in x]

    def download(self) -> None:
        """Download the raw data. This method should be overridden if you need to download the data."""
        pass

    def process(self) -> None:
        """Process the raw data and save it to the processed directory. This method should be overridden if you need to process the data."""
        pass

    def len(self) -> int:
        """Return the number of processed files.

        Returns:
            int: Number of processed files.
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """Get a processed data object.

        Args:
            idx (int): Index of the data object to retrieve.

        Returns:
            Data: The processed data object.
        """
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        data["tx"].x = data["tx"].x.to_dense()
        return data


def get_xy_extents(
    filepath,
    x: str,
    y: str,
) -> Tuple[int]:
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
    return x_min, y_min, x_max, y_max


def coo_to_dense_adj(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_nbrs: Optional[int] = None,
) -> torch.Tensor:

    # Check COO format
    if not edge_index.shape[0] == 2:
        msg = (
            "Edge index is not in COO format. First dimension should have " f"size 2, but found {edge_index.shape[0]}."
        )
        raise ValueError(msg)

    # Get split points
    uniques, counts = torch.unique(edge_index[0], return_counts=True)
    if num_nodes is None:
        num_nodes = uniques.max() + 1
    if num_nbrs is None:
        num_nbrs = counts.max()
    counts = tuple(counts.cpu().tolist())

    # Fill matrix with neighbors
    nbr_idx = torch.full((num_nodes, num_nbrs), -1)
    for i, nbrs in zip(uniques, torch.split(edge_index[1], counts)):
        nbr_idx[i, : len(nbrs)] = nbrs

    return nbr_idx


def format_time(elapsed: float) -> str:
    """
    Format elapsed time to h:m:s.

    Parameters:
    ----------
    elapsed : float
        Elapsed time in seconds.

    Returns:
    -------
    str
        Formatted time in h:m:s.
    """
    return str(timedelta(seconds=int(elapsed)))
