import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torchmetrics import F1Score
from scipy.sparse.csgraph import connected_components as cc

from segger.data.utils import get_edge_index_rapids
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from tqdm import tqdm
import random
import string
import cupy as cp  # Use CuPy for GPU-accelerated NumPy-like operations
from pathlib import Path
import glob
import re


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


# Optimized edge index computation using CuPy
def get_edge_index_cupy(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 5, dist: int = 10) -> cp.ndarray:
    """
    Compute edge index using CuPy to accelerate GPU operations.
    This replaces Dask's `parallel_get_edge_index`.
    """
    coords_1_cp = cp.asarray(coords_1)
    coords_2_cp = cp.asarray(coords_2)

    # Use RAPIDS or custom nearest neighbors logic (with CuPy)
    edge_index = get_edge_index_rapids(coords_1_cp, coords_2_cp, k=k, dist=dist)
    return edge_index


# Optimized similarity score computation using CuPy
def get_similarity_scores_cupy(model: Segger, batch: Batch, from_type: str, to_type: str):
    """
    Compute similarity scores using CuPy for sparse-dense matrix operations.

    Parameters
    ----------
    model : Segger
        The segmentation model used for similarity score computation.
    batch : Batch
        The batch of graph data containing transcript and boundary nodes.
    from_type : str
        The node type from which similarity is computed.
    to_type : str
        The node type to which similarity is computed.

    Returns
    -------
    torch.Tensor
        A dense tensor containing similarity scores between 'from_type' and 'to_type' nodes.
    """
    batch = batch.to("cuda")
    y_hat = model(batch.x_dict, batch.edge_index_dict)

    # Sparse adjacency matrix for neighbors
    sparse_adj = batch[from_type][f'{to_type}_field']
    
    # Perform sparse-dense multiplication (CuPy handles this on the GPU)
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


# Optimized predict_batch function using PyTorch parallelism (no Dask)
def predict_batch(lit_segger: LitSegger, batch: Batch, score_cut: float, receptive_field: dict, use_cc: bool = True) -> pd.DataFrame:
    """
    Predict cell assignments for a batch of transcript data using PyTorch parallelism.
    
    Parameters
    ----------
    lit_segger : LitSegger
        Lightning module wrapping the segmentation model.
    batch : Batch
        A batch of transcript and cell data.
    score_cut : float
        The threshold for assigning transcripts to cells based on similarity scores.
    receptive_field : dict
        Receptive field parameters for computing nearest neighbors.
    use_cc : bool
        Whether to use connected components for unassigned transcripts.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing transcript IDs, similarity scores, and assigned cell IDs.
    """
    def _get_id():
        id_chars = random.choices(string.ascii_lowercase, k=8)
        return ''.join(id_chars) + '-nx'

    with torch.no_grad():
        batch = batch.to("cuda")

        # Assignments of cells to nuclei
        assignments = pd.DataFrame()
        assignments['transcript_id'] = batch['tx'].id.cpu().numpy()

        if len(batch['bd'].id[0]) > 0:
            # Compute edge index using CuPy (GPU-accelerated)
            edge_index = get_edge_index_cupy(
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

            # Compute similarity scores using CuPy for GPU acceleration
            scores = get_similarity_scores_cupy(lit_segger.model, batch, "tx", "bd")

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

                scores = get_similarity_scores_cupy(lit_segger.model, batch, "tx", "tx")
                scores = scores.fill_diagonal_(0)

                # Connected components to assign remaining transcripts
                no_id = assignments['segger_cell_id'].isna().values
                no_id_scores = scores[no_id][:, no_id]
                n, comps = cc(no_id_scores, connection="weak", directed=False)
                new_ids = np.array([_get_id() for _ in range(n)])
                assignments.loc[no_id, 'segger_cell_id'] = new_ids[comps]

        return assignments


# Optimized predict function for multiple batches using CuPy
def predict(lit_segger: LitSegger, data_loader: DataLoader, score_cut: float, receptive_field: dict, use_cc: bool = True) -> pd.DataFrame:
    """
    Predict cell assignments for multiple batches using multi-GPU PyTorch.

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

    # Iterate through batches and run `predict_batch` for each
    for batch in tqdm(data_loader, desc="Processing batches"):
        batch_assignments = predict_batch(lit_segger, batch, score_cut, receptive_field, use_cc)
        assignments.append(batch_assignments)

    # Concatenate the results from all batches
    assignments = pd.concat(assignments).reset_index(drop=True)

    # Handle duplicate assignments of transcripts
    idx = assignments.groupby('transcript_id')['score'].idxmax()
    assignments = assignments.loc[idx].reset_index(drop=True)

    return assignments
