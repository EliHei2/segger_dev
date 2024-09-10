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
try_import('multiprocessing')
try_import('joblib')
try_import('faiss')
try_import('cuml')
try_import('cudf')
try_import('cugraph')
try_import('cuspatial')
try_import('hnswlib')




def filter_transcripts(
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
    )
    mask = transcripts_df["qv"].ge(min_qv)
    mask &= ~transcripts_df["feature_name"].str.startswith(filter_codewords)
    return transcripts_df[mask]


def compute_transcript_metrics(
    df: pd.DataFrame,
    qv_threshold: float = 30,
    cell_id_col: str = 'cell_id'
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
    df_filtered = df[df['qv'] > qv_threshold]
    total_transcripts = len(df_filtered)
    assigned_transcripts = df_filtered[df_filtered[cell_id_col] != -1]
    percent_assigned = len(assigned_transcripts) / total_transcripts * 100
    cytoplasmic_transcripts = assigned_transcripts[assigned_transcripts['overlaps_nucleus'] != 1]
    percent_cytoplasmic = len(cytoplasmic_transcripts) / len(assigned_transcripts) * 100
    percent_nucleus = 100 - percent_cytoplasmic
    non_assigned_transcripts = df_filtered[df_filtered[cell_id_col] == -1]
    non_assigned_cytoplasmic = non_assigned_transcripts[non_assigned_transcripts['overlaps_nucleus'] != 1]
    percent_non_assigned_cytoplasmic = len(non_assigned_cytoplasmic) / len(non_assigned_transcripts) * 100

    gene_group_assigned = assigned_transcripts.groupby('feature_name')
    gene_group_all = df_filtered.groupby('feature_name')
    gene_percent_assigned = (gene_group_assigned.size() / gene_group_all.size() * 100).reset_index(name='percent_assigned')
    cytoplasmic_gene_group = cytoplasmic_transcripts.groupby('feature_name')
    gene_percent_cytoplasmic = (cytoplasmic_gene_group.size() / len(cytoplasmic_transcripts) * 100).reset_index(name='percent_cytoplasmic')
    gene_metrics = pd.merge(gene_percent_assigned, gene_percent_cytoplasmic, on='feature_name', how='outer').fillna(0)

    results = {
        'percent_assigned': percent_assigned,
        'percent_cytoplasmic': percent_cytoplasmic,
        'percent_nucleus': percent_nucleus,
        'percent_non_assigned_cytoplasmic': percent_non_assigned_cytoplasmic,
        'gene_metrics': gene_metrics
    }
    return results


def create_anndata(
    df: pd.DataFrame, 
    panel_df: Optional[pd.DataFrame] = None, 
    min_transcripts: int = 5, 
    cell_id_col: str = 'cell_id', 
    qv_threshold: float = 30, 
    min_cell_area: float = 10.0, 
    max_cell_area: float = 1000.0
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
    df_filtered = filter_transcripts(df, min_qv=qv_threshold)
    metrics = compute_transcript_metrics(df_filtered, qv_threshold, cell_id_col)
    df_filtered = df_filtered[df_filtered[cell_id_col].astype(str) != '-1']
    pivot_df = df_filtered.rename(columns={
        cell_id_col: "cell",
        "feature_name": "gene"
    })[['cell', 'gene']].pivot_table(index='cell', columns='gene', aggfunc='size', fill_value=0)
    pivot_df = pivot_df[pivot_df.sum(axis=1) >= min_transcripts]
    cell_summary = []
    for cell_id, cell_data in df_filtered.groupby(cell_id_col):
        if len(cell_data) < min_transcripts:
            continue
        cell_convex_hull = ConvexHull(cell_data[['x_location', 'y_location']])
        cell_area = cell_convex_hull.area
        if cell_area < min_cell_area or cell_area > max_cell_area:
            continue
        if 'nucleus_distance' in cell_data:
            nucleus_data = cell_data[cell_data['nucleus_distance'] == 0]
        else:
            nucleus_data = cell_data[cell_data['overlaps_nucleus'] == 1]
        if len(nucleus_data) >= 3:
            nucleus_convex_hull = ConvexHull(nucleus_data[['x_location', 'y_location']])
        else:
            nucleus_convex_hull = None
        cell_summary.append({
            "cell": cell_id,
            "cell_centroid_x": cell_data['x_location'].mean(),
            "cell_centroid_y": cell_data['y_location'].mean(),
            "cell_area": cell_area,
            "nucleus_centroid_x": nucleus_data['x_location'].mean() if len(nucleus_data) > 0 else cell_data['x_location'].mean(),
            "nucleus_centroid_y": nucleus_data['x_location'].mean() if len(nucleus_data) > 0 else cell_data['x_location'].mean(),
            "nucleus_area": nucleus_convex_hull.area if nucleus_convex_hull else 0,
            "percent_cytoplasmic": len(cell_data[cell_data['overlaps_nucleus'] != 1]) / len(cell_data) * 100,
            "has_nucleus": len(nucleus_data) > 0
        })
    cell_summary = pd.DataFrame(cell_summary).set_index("cell")
    if panel_df is not None:
        panel_df = panel_df.sort_values('gene')
        genes = panel_df['gene'].values
        for gene in genes:
            if gene not in pivot_df:
                pivot_df[gene] = 0
        pivot_df = pivot_df[genes.tolist()]
    if panel_df is None:
        var_df = pd.DataFrame([{
            "gene": i, 
            "feature_types": 'Gene Expression', 
            'genome': 'Unknown'
        } for i in np.unique(pivot_df.columns.values)]).set_index('gene')
    else:
        var_df = panel_df[['gene', 'ensembl']].rename(columns={'ensembl':'gene_ids'})
        var_df['feature_types'] = 'Gene Expression'
        var_df['genome'] = 'Unknown'
        var_df = var_df.set_index('gene')
    gene_metrics = metrics['gene_metrics'].set_index('feature_name')
    var_df = var_df.join(gene_metrics, how='left').fillna(0)
    cells = list(set(pivot_df.index) & set(cell_summary.index))
    pivot_df = pivot_df.loc[cells,:]
    cell_summary = cell_summary.loc[cells,:]
    adata = ad.AnnData(pivot_df.values)
    adata.var = var_df
    adata.obs['transcripts'] = pivot_df.sum(axis=1).values
    adata.obs['unique_transcripts'] = (pivot_df > 0).sum(axis=1).values
    adata.obs_names = pivot_df.index.values.tolist()
    adata.obs = pd.merge(adata.obs, cell_summary.loc[adata.obs_names,:], left_index=True, right_index=True)
    adata.uns['metrics'] = {
        'percent_assigned': metrics['percent_assigned'],
        'percent_cytoplasmic': metrics['percent_cytoplasmic'],
        'percent_nucleus': metrics['percent_nucleus'],
        'percent_non_assigned_cytoplasmic': metrics['percent_non_assigned_cytoplasmic']
    }
    return adata

    

def calculate_gene_celltype_abundance_embedding(adata: ad.AnnData, celltype_column: str) -> pd.DataFrame:
    """Calculate the cell type abundance embedding for each gene based on the percentage of cells in each cell type 
    that express the gene (non-zero expression).

    Parameters:
        adata (ad.AnnData): An AnnData object containing gene expression data and cell type information.
        celltype_column (str): The column name in `adata.obs` that contains the cell type information.

    Returns:
        pd.DataFrame: A DataFrame where rows are genes and columns are cell types, with each value representing 
            the percentage of cells in that cell type expressing the gene.
            
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
    # Calculate the percentage of cells expressing each gene per cell type
    cell_type_abundance_list = []
    for i in range(cell_type_encoded.shape[1]):
        # Extract cells of the current cell type
        cell_type_mask = cell_type_encoded[:, i] == 1
        # Calculate the abundance: sum of non-zero expressions in this cell type / total cells in this cell type
        abundance = gene_expression_df[cell_type_mask].mean(axis=0) * 100
        cell_type_abundance_list.append(abundance)
    # Create a DataFrame for the cell type abundance with gene names as rows and cell types as columns
    cell_type_abundance_df = pd.DataFrame(cell_type_abundance_list, 
                                            columns=adata.var_names, 
                                            index=encoder.categories_[0]).T
    return cell_type_abundance_df

def get_edge_index(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10, method: str = 'kd_tree',
                   gpu: bool = False, workers: int = 1) -> torch.Tensor:
    """
    Computes edge indices using various methods (KD-Tree, FAISS, RAPIDS cuML, cuGraph, or cuSpatial).

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.
        method (str, optional): The method to use ('kd_tree', 'faiss', 'rapids', 'cugraph', 'cuspatial').
        gpu (bool, optional): Whether to use GPU acceleration (applicable for FAISS).

    Returns:
        torch.Tensor: Edge indices.
    """
    if method == 'kd_tree':
        return get_edge_index_kdtree(coords_1, coords_2, k=k, dist=dist, workers=workers)
    elif method == 'faiss':
        return get_edge_index_faiss(coords_1, coords_2, k=k, dist=dist, gpu=gpu)
    elif method == 'rapids':
        return get_edge_index_rapids(coords_1, coords_2, k=k, dist=dist)
    elif method == 'cugraph':
        return get_edge_index_cugraph(coords_1, coords_2, k=k, dist=dist)
    elif method == 'cuspatial':
        return get_edge_index_cuspatial(coords_1, coords_2, k=k, dist=dist)
    elif method == 'hnsw':
        return get_edge_index_hnsw(coords_1, coords_2, k=k, dist=dist)
    else:
        raise ValueError(f"Unknown method {method}")



def get_edge_index_kdtree(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10, workers: int = 1) -> torch.Tensor:
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
    tree = cKDTree(coords_1)
    d_kdtree, idx_out = tree.query(coords_2, k=k, distance_upper_bound=dist, workers=workers)
    valid_mask = d_kdtree < dist
    edges = []

    for idx, valid in enumerate(valid_mask):
        valid_indices = idx_out[idx][valid]
        if valid_indices.size > 0:
            edges.append(
                np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T
            )

    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index


def get_edge_index_faiss(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10, gpu: bool = False) -> torch.Tensor:
    """
    Computes edge indices using FAISS.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.
        gpu (bool, optional): Whether to use GPU acceleration.

    Returns:
        torch.Tensor: Edge indices.
    """
    coords_1 = np.ascontiguousarray(coords_1, dtype=np.float32)
    coords_2 = np.ascontiguousarray(coords_2, dtype=np.float32)
    d = coords_1.shape[1]
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(coords_1.astype('float32'))
    D, I = index.search(coords_2.astype('float32'), k)

    valid_mask = D < dist ** 2
    edges = []

    for idx, valid in enumerate(valid_mask):
        valid_indices = I[idx][valid]
        if valid_indices.size > 0:
            edges.append(
                np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T
            )

    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index


def get_edge_index_rapids(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor:
    """
    Computes edge indices using RAPIDS cuML.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.

    Returns:
        torch.Tensor: Edge indices.
    """
    index = cuml.neighbors.NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    index.fit(coords_1)
    D, I = index.kneighbors(coords_2)

    valid_mask = D < dist ** 2
    edges = []

    for idx, valid in enumerate(valid_mask):
        valid_indices = I[idx][valid]
        if valid_indices.size > 0:
            edges.append(
                np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T
            )

    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index

def get_edge_index_cugraph(
    coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10
) -> torch.Tensor:
    """
    Computes edge indices using RAPIDS cuGraph.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.

    Returns:
        torch.Tensor: Edge indices.
    """
    gdf_1 = cudf.DataFrame({'x': coords_1[:, 0], 'y': coords_1[:, 1]})
    gdf_2 = cudf.DataFrame({'x': coords_2[:, 0], 'y': coords_2[:, 1]})

    gdf_1['id'] = gdf_1.index
    gdf_2['id'] = gdf_2.index

    result = cugraph.spatial_knn(
        gdf_1, gdf_2, k=k, return_distance=True
    )

    valid_mask = result['distance'] < dist
    edges = result[['src', 'dst']].loc[valid_mask].to_pandas().values
    edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
    return edge_index


def get_edge_index_cuspatial(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor:
    """
    Computes edge indices using cuSpatial's spatial join functionality.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates (2D).
        coords_2 (np.ndarray): Second set of coordinates (2D).
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.

    Returns:
        torch.Tensor: Edge indices.
    """
    # Convert numpy arrays to cuDF DataFrames
    coords_1_df = cudf.DataFrame({'x': coords_1[:, 0], 'y': coords_1[:, 1]})
    coords_2_df = cudf.DataFrame({'x': coords_2[:, 0], 'y': coords_2[:, 1]})
    
    # Perform the nearest neighbor search using cuSpatial's point-to-point nearest neighbor
    result = cuspatial.point_to_nearest_neighbor(
        coords_1_df['x'], coords_1_df['y'],
        coords_2_df['x'], coords_2_df['y'],
        k=k
    )
    
    # The result is a tuple (distances, indices)
    distances, indices = result
    
    # Filter by distance threshold
    valid_mask = distances < dist
    edges = []
    
    for idx, valid in enumerate(valid_mask):
        valid_indices = indices[idx][valid]
        if valid_indices.size > 0:
            edges.append(
                np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T
            )
    
    # Convert to torch.Tensor
    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index




def get_edge_index_hnsw(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor:
    """
    Computes edge indices using the HNSW algorithm.

    Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int, optional): Number of nearest neighbors.
        dist (int, optional): Distance threshold.

    Returns:
        torch.Tensor: Edge indices.
    """
    num_elements = coords_1.shape[0]
    dim = coords_1.shape[1]

    # Initialize the HNSW index
    p = hnswlib.Index(space='l2', dim=dim)  # l2 for Euclidean distance
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Add points to the index
    p.add_items(coords_1)

    # Query the index for nearest neighbors
    indices, distances = p.knn_query(coords_2, k=k)

    # Filter by distance threshold
    valid_mask = distances < dist ** 2
    edges = []

    for idx, valid in enumerate(valid_mask):
        valid_indices = indices[idx][valid]
        if valid_indices.size > 0:
            edges.append(
                np.vstack((np.full(valid_indices.shape, idx), valid_indices)).T
            )

    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long).contiguous()
    return edge_index

class SpatialTranscriptomicsDataset(InMemoryDataset):
    """A dataset class for handling SpatialTranscriptomics spatial transcriptomics data.

    Attributes:
        root (str): The root directory where the dataset is stored.
        transform (callable): A function/transform that takes in a Data object and returns a transformed version.
        pre_transform (callable): A function/transform that takes in a Data object and returns a transformed version.
        pre_filter (callable): A function that takes in a Data object and returns a boolean indicating whether to keep it.
    """
    def __init__(self, root: str, transform: Callable = None, pre_transform: Callable = None, pre_filter: Callable = None):
        """Initialize the SpatialTranscriptomicsDataset.

        Args:
            root (str): Root directory where the dataset is stored.
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version. Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version. Defaults to None.
            pre_filter (callable, optional): A function that takes in a Data object and returns a boolean indicating whether to keep it. Defaults to None.
        """
        super().__init__(root, transform, pre_transform, pre_filter)
        os.makedirs(os.path.join(self.processed_dir, 'raw'), exist_ok=True)

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
        return [x for x in os.listdir(self.processed_dir) if 'tiles' in x]

    def download(self) -> None:
        """Download the raw data. This method should be overridden if you need to download the data.
        """
        pass

    def process(self) -> None:
        """Process the raw data and save it to the processed directory. This method should be overridden if you need to process the data.
        """
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
        data['tx'].x = data['tx'].x.to_dense()
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