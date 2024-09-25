import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torchmetrics import F1Score
from scipy.sparse.csgraph import connected_components as cc

from segger.data.utils import (
    get_edge_index_rapids,
    coo_to_dense_adj,
)
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from tqdm import tqdm
import random
import string
import cupy as cp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import dask.array as da
import re

# Function to set up the Dask-CUDA cluster with a configurable number of GPUs
def initialize_dask_cuda_cluster(num_gpus: int):
    """
    Initialize a Dask-CUDA cluster with a specific number of GPUs.

    Parameters
    ----------
    num_gpus : int
        Number of GPUs to use in the Dask-CUDA cluster.
    """
    # Initialize a LocalCUDACluster with the number of desired GPUs
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=list(range(num_gpus)))
    client = Client(cluster)
    return client

# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Function to load the model checkpoint
def load_model(checkpoint_path: str) -> LitSegger:
    checkpoint_path = Path(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        checkpoints = glob.glob(str(checkpoint_path / '*.ckpt'))
        def sort_order(c):
            match = re.match(r'.*epoch=(\d+)-step=(\d+).ckpt', c)
            return int(match[1]), int(match[2])
        checkpoint_path = Path(sorted(checkpoints, key=sort_order)[-1])

    lit_segger = LitSegger.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return lit_segger

# Optimized edge index computation using Dask and CuPy (with RAPIDS)
def parallel_get_edge_index(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10):
    coords_1_cp = cp.asarray(coords_1)
    coords_2_cp = cp.asarray(coords_2)

    def compute_edges_chunk(coords_1_chunk, coords_2_chunk):
        return get_edge_index_rapids(coords_1_chunk, coords_2_chunk, k=k, dist=dist)

    coords_1_da = da.from_array(coords_1_cp, chunks=(len(coords_1_cp) // 4, 2))
    coords_2_da = da.from_array(coords_2_cp, chunks=(len(coords_2_cp) // 4, 2))

    edge_indices = da.map_blocks(compute_edges_chunk, coords_1_da, coords_2_da).compute()

    return edge_indices

# Optimized similarity score computation using Dask and CuPy
def parallel_get_similarity_scores(client: Client, model, batch_chunks, from_type, to_type):
    """
    Compute similarity scores in parallel using Dask and CuPy.
    
    Parameters:
    ----------
    client : Client
        Dask distributed client to manage parallel computation.
    model : Segger
        The segmentation model used for similarity score computation.
    batch_chunks : list
        A list of batches to process in parallel.
    from_type : str
        The type of node from which the similarity is computed.
    to_type : str
        The type of node to which the similarity is computed.
    
    Returns:
    -------
    list
        A list of similarity score tensors, one per batch.
    """
    def compute_similarity(batch_chunk):
        return get_similarity_scores(model, batch_chunk, from_type, to_type)

    # Distribute the batch_chunks to the Dask client
    similarity_results = client.map(compute_similarity, batch_chunks)
    
    # Gather the results back from all workers
    results = client.gather(similarity_results)
    
    return results

# Optimized similarity score computation with sparse matrix operations
def get_similarity_scores(model: Segger, batch: Batch, from_type: str, to_type: str):
    batch = batch.to("cuda")
    y_hat = model(batch.x_dict, batch.edge_index_dict)

    # Sparse adjacency matrix for neighbors
    sparse_adj = batch[from_type][f'{to_type}_field']
    
    # Perform sparse-dense multiplication
    similarity = torch.sparse.mm(
        sparse_adj, 
        y_hat[to_type]
    )

    # Compute similarity between neighbors
    similarity = torch.bmm(
        similarity.unsqueeze(1),  # Add dimension for batch matrix multiplication
        y_hat[from_type].unsqueeze(-1)  # 'from_type' embedding
    ).squeeze(-1)

    similarity[similarity == 0] = -torch.inf  # Ensure zero stays zero
    similarity = F.sigmoid(similarity)

    indices = torch.nonzero(batch[from_type][f'{to_type}_field'].to_dense().long() != -1).T
    values = similarity[batch[from_type][f'{to_type}_field'].to_dense().long() != -1].flatten()

    sparse_sim = torch.sparse_coo_tensor(indices, values, (batch[from_type].x.shape[0], batch[to_type].x.shape[0]))

    return sparse_sim.to_dense().detach().cpu()

# Optimized predict_batch with Dask and CuPy, multi-GPU support
def predict_batch(lit_segger: LitSegger, batch: Batch, score_cut: float, receptive_field: dict, use_cc: bool = True, num_gpus: int = 1) -> pd.DataFrame:
    def _get_id():
        id_chars = random.choices(string.ascii_lowercase, k=8)
        return ''.join(id_chars) + '-nx'
    
    client = initialize_dask_cuda_cluster(num_gpus)  # Initialize Dask with multiple GPUs
    
    with torch.no_grad():
        batch = batch.to("cuda")

        # Assignments of cells to nuclei
        assignments = pd.DataFrame()
        assignments['transcript_id'] = batch['tx'].id.cpu().numpy()

        if len(batch['bd'].id[0]) > 0:
            # Parallel edge index computation using Dask and CuPy
            edge_index = parallel_get_edge_index(
                batch['bd'].pos[:, :2].cpu().numpy(),
                batch['tx'].pos[:, :2].cpu().numpy(),
                k=receptive_field['k_bd'],
                dist=receptive_field['dist_bd']
            ).T

            # Sparse adjacency matrix (optimized, no dense conversion)
            batch['tx']['bd_field'] = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.shape[1], device=edge_index.device),
                size=(batch['tx'].id.shape[0], batch['bd'].id.shape[0])
            ).coalesce()

            # Compute similarity scores using Dask for parallel computation
            scores = parallel_get_similarity_scores(client, lit_segger.model, [batch], "tx", "bd")[0]

            # Get direct assignments from similarity matrix
            belongs = scores.max(1)
            assignments['score'] = belongs.values.cpu()

            # Filter transcripts that meet the score cutoff
            mask = assignments['score'] > score_cut
            all_ids = np.concatenate(batch['bd'].id)[belongs.indices.cpu()]
            assignments.loc[mask, 'segger_cell_id'] = all_ids[mask]

            if use_cc:
                # Transcript-transcript similarity using sparse matrices
                edge_index = batch['tx', 'neighbors', 'tx'].edge_index
                batch['tx']['tx_field'] = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.shape[1], device=edge_index.device),
                    size=(batch['tx'].id.shape[0], batch['tx'].id.shape[0])
                ).coalesce()

                scores = get_similarity_scores(lit_segger.model, batch, "tx", "tx")
                scores = scores.fill_diagonal_(0)

                # Connected components to assign remaining transcripts
                no_id = assignments['segger_cell_id'].isna().values
                no_id_scores = scores[no_id][:, no_id]
                n, comps = cc(no_id_scores, connection="weak", directed=False)
                new_ids = np.array([_get_id() for _ in range(n)])
                assignments.loc[no_id, 'segger_cell_id'] = new_ids[comps]

        return assignments

# Optimized predict function that processes multiple batches in parallel
def predict(lit_segger: LitSegger, data_loader: DataLoader, score_cut: float, receptive_field: dict, use_cc: bool = True, num_gpus: int = 1) -> pd.DataFrame:
    """
    Predict cell assignments for multiple batches using multi-GPU Dask processing.

    Parameters
    ----------
    lit_segger : LitSegger
        The lightning module wrapping the segmentation model.
    data_loader : DataLoader
        A data loader providing batches of transcript and cell data.
    score_cut : float
        The threshold for assigning transcripts to cells based on similarity scores.
    receptive_field : dict
        Dictionary specifying the receptive field for boundary detection.
    use_cc : bool
        Whether to use connected components for remaining assignments.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the transcript IDs, similarity scores, and assigned cell IDs.
    """
    if len(data_loader) == 0:
        return None
    
    assignments = []

    # Use Dask-CUDA to distribute the batches across multiple GPUs
    client = initialize_dask_cuda_cluster(num_gpus)

    # Distribute batches across GPUs
    futures = client.map(lambda batch: predict_batch(lit_segger, batch, score_cut, receptive_field, use_cc, num_gpus), list(data_loader))
    results = client.gather(futures)
    
    # Collect assignments from all batches
    assignments = pd.concat(results).reset_index(drop=True)

    # Handle duplicate assignments of transcripts
    idx = assignments.groupby('transcript_id')['score'].idxmax()
    assignments = assignments.loc[idx].reset_index(drop=True)

    return assignments
