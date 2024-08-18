import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torchmetrics import F1Score
from scipy.sparse.csgraph import connected_components as cc

from segger.data.utils import XeniumDataset
from segger.data.utils import XeniumSample
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from lightning import LightningModule
from torch_geometric.nn import to_hetero
import random
import string
import os
import yaml
from pathlib import Path
import glob
import typing
import re


# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_model(
    checkpoint_path: os.PathLike,
) -> LitSegger:
    """
    Load a LitSegger model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : os.Pathlike
        Specific checkpoint file to load, or directory where the model 
        checkpoints are stored. If directory, the latest checkpoint is loaded.

    Returns
    -------
    LitSegger
        The loaded LitSegger model.

    Raises
    ------
    FileNotFoundError
        If the specified checkpoint file does not exist.
    """
    # Get last checkpoint if directory provided
    checkpoint_path = Path(checkpoint_path)
    msg = (
        f"No checkpoint found at {checkpoint_path}. Please make sure "
        "you've provided the correct path."
    )
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

    # Load model
    lit_segger = LitSegger.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device("cuda"),
    )

    return lit_segger


def get_similarity_scores(
    model: Segger, 
    batch: Batch,
    from_type: str,
    to_type: str,
):
    """
    Computes similarity scores between 'from_type' and 'to_type' nodes.

    Parameters
    ----------
    model : Segger
        The Segger model to be used for computing embeddings.
    batch : Batch
        The batch of data containing node features and edge indices.
    from_type : str
        The type of nodes from which similarity is computed.
    to_type : str
        The type of nodes to which similarity is computed.

    Returns
    -------
    scores : torch.Tensor
        A tensor containing the similarity scores between 'from_type' and 
        'to_type' nodes.

    Notes
    -----
    This function computes the similarity scores by obtaining the embedding 
    spaces from the model, padding the embeddings, and then calculating the 
    similarity of each 'from_type' node to its 'to_type' neighbors in the 
    embedding space. The similarity scores are then filtered and returned.
    """

    # Get embedding spaces from model
    batch = batch.to("cuda")
    y_hat = model(batch.x_dict, batch.edge_index_dict)
    m = torch.nn.ZeroPad2d((0, 0, 0, 1))  # pad bottom with zeros
    y_hat[to_type] = m(y_hat[to_type])

    # Similarity of each 'from_type' to 'to_type' neighbors in embedding
    nbr_idx = batch[from_type][f'{to_type}_field']
    similarity = torch.bmm(
        y_hat[to_type][nbr_idx],       # 'to' x 'from' neighbors x embed
        y_hat[from_type].unsqueeze(-1) # 'to' x embed x 1
    )                                  # -> 'to' x 'from' neighbors x 1

    # Softmax to get most similar 'to_type' neighbor
    similarity[similarity == 0] = -torch.inf  # ensure zero stays zero
    similarity = F.sigmoid(similarity)

    # Sparse adjacency indices from neighbors
    shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0] + 1
    adj = XeniumSample.kd_to_edge_index_(shape, nbr_idx.cpu().detach()).cpu()
    adj = adj[:, adj[0, :].argsort()]  # sort to correct indexing order

    # Neighbor-filtered similarity scores
    scores = torch.zeros(shape)
    scores[adj[0], adj[1]] = similarity.flatten().cpu()

    return scores


def predict(
    lit_segger: LitSegger,
    data_loader: DataLoader,
    score_cut: float,
    k_nc: int,
    dist_nc: int,
    k_tx: int,
    dist_tx: int,
) -> pd.DataFrame:
    """
    """
    with torch.no_grad():
        
        for batch in data_loader:

            batch = batch.to("cuda")

            # Transcript-cell similarity scores, filtered by neighbors
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "nc")
            belongs = scores.max(1)

            # Get direct assignments of transcripts to cells
            ids_tx = (
                np.concatenate(batch["tx"].id).reshape((1, -1))[0].astype("str")
            )
            ids_nc = (
                np.concatenate(batch["nc"].id).reshape((1, -1))[0].astype("str")
            )
            belongs.indices[belongs.indices == len(ids_nc)] = 0
            mapping = np.vstack(
                (ids_tx, ids_nc[belongs.indices], belongs.values)
            )
            mapping[1, np.where(belongs.values < score_cut)] = "UNASSIGNED"
            mapping[1, np.where(belongs.values == 0)] = "FLOATING"

            idx_unknown = np.where(
                (mapping[1, :] == "UNASSIGNED") | (mapping[1, :] == "FLOATING")
            )[0]
            ids_unknown = ids_tx[idx_unknown]
            
            # Transcript-transcript similarity scores, filtered by neighbors
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "tx")
            f_output = scores

            f_output = f_output.fill_diagonal_(0)
            temp = torch.zeros((len(idx_unknown), len(idx_unknown))).cpu()
            temp[: len(idx_unknown), : len(idx_unknown)] = f_output[
                idx_unknown, :
            ][:, idx_unknown]
            n_comps, comps = cc(temp, connection="weak", directed=False)
            random_strings = np.array(
                [
                    "".join(random.choices(string.ascii_lowercase, k=8)) + "-nx"
                    for _ in range(n_comps)
                ]
            )
            new_ids = random_strings[comps]
            new_mapping = np.vstack(
                (ids_unknown, new_ids, np.zeros(len(new_ids)))
            )
            mapping = mapping[
                :,
                ~(
                    (mapping[1, :] == "UNASSIGNED")
                    | (mapping[1, :] == "FLOATING")
                ),
            ]
            mapping = np.hstack((mapping, new_mapping))

            if all_mappings is None:
                all_mappings = mapping
            else:
                all_mappings = np.hstack((all_mappings, mapping))

    mappings_df = pd.DataFrame(
        all_mappings.T, columns=["transcript_id", "segger_cell_id", "score"]
    )
    idx = mappings_df.groupby("transcript_id")["score"].idxmax()
    mappings_df = mappings_df.loc[idx].reset_index(drop=True)
    return mappings_df
