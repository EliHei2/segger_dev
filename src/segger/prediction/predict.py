import os
import torch
import dask.dataframe as dd
import pandas as pd
import torch.nn.functional as F
from dask import delayed
from tqdm import tqdm
if torch.cuda.is_available():
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client, LocalCluster
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc
from torch_geometric.utils import to_scipy_sparse_matrix
from path import Path
import glob 
from dask.diagnostics import ProgressBar
from typing import Dict
import dask

from segger.data.utils import (
    SpatialTranscriptomicsDataset,
    get_edge_index,
    coo_to_dense_adj,
)
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
import re
import random
import numpy as np

# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up Dask cluster for distributed GPU computation (optional)
if torch.cuda.is_available():
    cluster = LocalCUDACluster()
    client = Client(cluster)

def load_model(checkpoint_path: str) -> LitSegger:
    """
    Load a LitSegger model from a checkpoint file.

    Parameters:
        checkpoint_path (str): 
            The file path to a specific checkpoint file or a directory where 
            model checkpoints are stored. If a directory, the latest checkpoint is loaded.

    Returns:
        LitSegger: The loaded LitSegger model.

    Raises:
        FileNotFoundError: If no checkpoint is found in the given directory.
        FileExistsError: If the given file path doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    msg = f"No checkpoint found at {checkpoint_path}."

    if os.path.isdir(checkpoint_path):
        checkpoints = glob.glob(str(checkpoint_path / '*.ckpt'))
        if len(checkpoints) == 0:
            raise FileNotFoundError(msg)
        def sort_order(c):
            match = re.match(r'.*epoch=(\d+)-step=(\d+).ckpt', c)
            return int(match[1]), int(match[2])
        checkpoint_path = Path(sorted(checkpoints, key=sort_order)[-1])
    elif not checkpoint_path.exists():
        raise FileExistsError(msg)

    lit_segger = LitSegger.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )

    return lit_segger


def get_similarity_scores(model: Segger, batch: Batch, from_type: str, to_type: str, device: 'str' = 'cuda') -> torch.Tensor:
    """
    Compute similarity scores between 'from_type' and 'to_type' embeddings within a batch.

    Parameters:
        model (Segger): The segmentation model used to generate embeddings.
        batch (Batch): A batch of data containing input features and edge indices.
        from_type (str): The type of node from which the similarity is computed.
        to_type (str): The type of node to which the similarity is computed.

    Returns:
        torch.Tensor: A dense tensor containing the similarity scores between 'from_type' 
                      and 'to_type' nodes.
    """
    batch = batch.to(device)
    y_hat = model(batch.x_dict, batch.edge_index_dict)

    # Similarity of each 'from_type' to 'to_type' neighbors in embedding
    nbr_idx = batch[from_type][f'{to_type}_field']
    m = torch.nn.ZeroPad2d((0, 0, 0, 1))  # pad bottom with zeros
    similarity = torch.bmm(
        m(y_hat[to_type])[nbr_idx],    # 'to' x 'from' neighbors x embed
        y_hat[from_type].unsqueeze(-1) # 'to' x embed x 1
    )

    # Sigmoid to get most similar 'to_type' neighbor
    similarity[similarity == 0] = -torch.inf  # ensure zero stays zero
    similarity = F.sigmoid(similarity)

    # Neighbor-filtered similarity scores
    shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]
    indices = torch.argwhere(nbr_idx != -1).T
    indices[1] = nbr_idx[nbr_idx != -1]
    values = similarity[nbr_idx != -1].flatten()
    sparse_sim = torch.sparse_coo_tensor(indices, values, shape)

    # Return in dense format
    scores = sparse_sim.to_dense().detach().cpu()
    return scores


def predict_batch(
    lit_segger: LitSegger, 
    batch: Batch, 
    score_cut: float, 
    receptive_field: dict, 
    use_cc: bool = True,
    device: 'str' = 'cuda'
) -> pd.DataFrame:
    """
    Predict cell assignments for a batch of transcript data using a segmentation model.

    Parameters:
        lit_segger (LitSegger): 
            The lightning module wrapping the segmentation model.
        batch (Batch): 
            A batch of transcript and cell data.
        score_cut (float): 
            The threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (dict): 
            A dictionary specifying the receptive field settings.
        use_cc (bool, optional): 
            Whether to use connected components for unassigned transcripts. Default is True.

    Returns:
        pd.DataFrame: A DataFrame containing the transcript IDs, similarity scores, 
                      and assigned cell IDs.
    """
    def _get_id():
        id_chars = random.choices(string.ascii_lowercase, k=8)
        return ''.join(id_chars) + '-nx'

    with torch.no_grad():
        batch = batch.to(device)
        assignments = pd.DataFrame()
        assignments['transcript_id'] = batch['tx'].id.cpu().numpy()

        if len(batch['bd'].id[0]) > 0:
            # Transcript-cell similarity scores
            edge_index = get_edge_index(
                batch['bd'].pos[:, :2].cpu(),
                batch['tx'].pos[:, :2].cpu(),
                k=receptive_field['k_bd'],
                dist=receptive_field['dist_bd'],
                method='kd_tree',
            ).T
            batch['tx']['bd_field'] = coo_to_dense_adj(
                edge_index,
                num_nodes=batch['tx'].id.shape[0],
                num_nbrs=receptive_field['k_bd'],
            )
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd", device=device)
            belongs = scores.max(1)
            assignments['score'] = belongs.values.cpu()
            mask = assignments['score'] > score_cut
            all_ids = np.concatenate(batch['bd'].id)[belongs.indices.cpu()]
            assignments.loc[mask, 'segger_cell_id'] = all_ids[mask]

            if use_cc:
                # Transcript-transcript similarity scores
                edge_index = batch['tx', 'neighbors', 'tx'].edge_index
                batch['tx']['tx_field'] = coo_to_dense_adj(
                    edge_index,
                    num_nodes=batch['tx'].id.shape[0],
                )
                scores = get_similarity_scores(lit_segger.model, batch, "tx", "tx", device=device)
                scores = scores.fill_diagonal_(0)

                # Sparse connected components for unassigned transcripts using SciPy
                no_id = assignments['segger_cell_id'].isna().values
                no_id_scores = scores[no_id][:, no_id]
                
                # Convert the similarity scores to a sparse matrix for SciPy
                sparse_matrix = csr_matrix(no_id_scores.numpy())
                n, comps = cc(sparse_matrix, connection="weak", directed=False)
                
                # Assign new IDs based on connected components
                new_ids = np.array([_get_id() for _ in range(n)])
                assignments.loc[no_id, 'segger_cell_id'] = new_ids[comps]

        return assignments



def predict(
    lit_segger: LitSegger, 
    data_loader: DataLoader, 
    score_cut: float, 
    receptive_field: dict, 
    use_cc: bool = True,
    device: str = 'cuda',
    num_workers: int = 4,
    k_gpus: int = None
) -> dd.DataFrame:
    """
    Predict cell assignments for multiple batches of transcript data using Dask for parallel processing.

    Parameters:
        lit_segger (LitSegger): 
            The lightning module wrapping the segmentation model.
        data_loader (DataLoader): 
            A data loader providing batches of transcript and cell data.
        score_cut (float): 
            The threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (dict): 
            A dictionary specifying the receptive field settings.
        use_cc (bool, optional): 
            Whether to use connected components for unassigned transcripts. Default is True.
        device (str, optional): 
            The device to run the model on, either 'cuda' (GPU) or 'cpu' (default).
        num_workers (int, optional): 
            The number of parallel workers to use for batch processing. Default is 4 for CPU.
        k_gpus (int, optional): 
            The number of GPUs to use when `device='cuda'`. Default is None, in which case the system will use all available GPUs.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the transcript IDs, similarity scores, 
                      and assigned cell IDs, consolidated across all batches.
    """
    # Step 1: Choose the cluster based on the device (CPU or GPU)
    if device == 'cpu':
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        client = Client(cluster)
    elif device == 'cuda':
        if k_gpus is None:
            k_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1 if "CUDA_VISIBLE_DEVICES" in os.environ else 1
        cluster = LocalCUDACluster(n_workers=k_gpus, threads_per_worker=1)
        client = Client(cluster)
    else:
        raise ValueError(f"Unknown device '{device}', must be either 'cpu' or 'cuda'.")

    # Step 2: Scatter the model (lit_segger) to all workers
    scattered_model = client.scatter(lit_segger, broadcast=True)
    
    # Step 3: Create a list of futures using Dask's `client.submit`
    futures = [
        client.submit(predict_batch, scattered_model, batch, score_cut, receptive_field, use_cc, device)
        for batch in tqdm(data_loader)
    ]
    
    # Step 4: Gather the results (futures) from the workers
    delayed_batches = [delayed(f.result()) for f in futures]

    # Step 5: Combine all delayed results into a single Dask DataFrame
    dask_assignments = dd.from_delayed(delayed_batches)

    # Step 6: Sort the DataFrame by 'transcript_id' for proper indexing
    dask_assignments = dask_assignments.sort_values('transcript_id')

    # Step 7: Set 'transcript_id' as the index and ensure the divisions are known
    dask_assignments = dask_assignments.set_index('transcript_id', sorted=True)

    # Step 8: Compute the index of maximum scores
    idx = dask_assignments.groupby('transcript_id')['score'].idxmax().compute()

    # Step 9: Use the computed index to filter the assignments
    final_assignments = dask_assignments.loc[idx]

    # Step 10: Execute the computation with a progress bar
    with ProgressBar():
        result = dask.compute(final_assignments)[0]

    # Shutdown the Dask client and cluster after execution
    client.close()
    cluster.close()

    return result