import os
import torch
import dask.dataframe as dd
import pandas as pd
import torch.nn.functional as F
from dask import delayed
from tqdm import tqdm
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc
from torch_geometric.utils import to_scipy_sparse_matrix

from segger.data.utils import (
    SpatialTranscriptomicsDataset,
    get_edge_index,
    coo_to_dense_adj,
)
from segger.models.segger_model import Segger
from segger.training.train import LitSegger

# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up Dask cluster for distributed GPU computation (optional)
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


def get_similarity_scores(model: Segger, batch: Batch, from_type: str, to_type: str) -> torch.Tensor:
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
    batch = batch.to("cuda")
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
    use_cc: bool = True
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
        batch = batch.to("cuda")
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
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd")
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
                scores = get_similarity_scores(lit_segger.model, batch, "tx", "tx")
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
    use_cc: bool = True
) -> dd.DataFrame:
    """
    Predict cell assignments for multiple batches of transcript data using Dask.

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

    Returns:
        dd.DataFrame: A Dask DataFrame containing the transcript IDs, similarity scores, 
                      and assigned cell IDs, consolidated across all batches.
    """
    if len(data_loader) == 0:
        return None

    delayed_batches = []
    
    # Process batches in parallel using Dask delayed
    for batch in tqdm(data_loader):
        delayed_batch = delayed(predict_batch)(
            lit_segger, batch, score_cut, receptive_field, use_cc
        )
        delayed_batches.append(delayed_batch)

    # Combine all delayed results into a single Dask DataFrame
    dask_assignments = dd.from_delayed(delayed_batches)

    # Handle duplicate assignments of transcripts
    idx = dask_assignments.groupby('transcript_id')['score'].idxmax()
    final_assignments = dask_assignments.loc[idx].compute()

    return final_assignments
