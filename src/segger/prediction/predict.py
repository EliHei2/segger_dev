import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
from torch_geometric.data import Batch
from scipy.sparse.csgraph import connected_components as cc
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
import random
import string
from pathlib import Path
import glob
from typing import Optional
import re
from scipy.spatial import KDTree


# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_model(checkpoint_path: str) -> LitSegger:
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
        checkpoints = glob.glob(str(checkpoint_path / "*.ckpt"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(msg)

        def sort_order(c):
            match = re.match(r".*epoch=(\d+)-step=(\d+).ckpt", c)
            return int(match[1]), int(match[2])

        checkpoint_path = Path(sorted(checkpoints, key=sort_order)[-1])
    elif not checkpoint_path.exists():
        raise FileExistsError(msg)

    # Load model
    lit_segger = LitSegger.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        # map_location=torch.device("cuda"),
    )

    return lit_segger


def get_similarity_scores(
    model: Segger,
    batch: Batch,
    from_type: str,
    to_type: str,
):
    """
    Compute similarity scores between 'from_type' and 'to_type' embeddings
    within a batch.

    Parameters
    ----------
    model : Segger
        The segmentation model used to generate embeddings.
    batch : Batch
        A batch of data containing input features and edge indices.
    from_type : str
        The type of node from which the similarity is computed.
    to_type : str
        The type of node to which the similarity is computed.

    Returns
    -------
    torch.Tensor
        A dense tensor containing the similarity scores between 'from_type'
        and 'to_type' nodes.
    """
    # Get embedding spaces from model
    batch = batch.to("cuda")
    y_hat = model(batch.x_dict, batch.edge_index_dict)

    # Similarity of each 'from_type' to 'to_type' neighbors in embedding
    nbr_idx = batch[from_type][f"{to_type}_field"]
    m = torch.nn.ZeroPad2d((0, 0, 0, 1))  # pad bottom with zeros
    similarity = torch.bmm(
        m(y_hat[to_type])[nbr_idx],  # 'to' x 'from' neighbors x embed
        y_hat[from_type].unsqueeze(-1),  # 'to' x embed x 1
    )  # -> 'to' x 'from' neighbors x 1

    # Sigmoid to get most similar 'to_type' neighbor
    similarity[similarity == 0] = -torch.inf  # ensure zero stays zero
    similarity = F.sigmoid(similarity)

    # Neighbor-filtered similarity scores
    shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]
    indices = torch.argwhere(nbr_idx != -1).T
    indices[1] = nbr_idx[nbr_idx != -1]
    values = similarity[nbr_idx != -1].flatten()
    sparse_sim = torch.sparse_coo_tensor(indices, values, shape)

    # Return in dense format for backwards compatibility
    scores = sparse_sim.to_dense().detach().cpu()

    return scores


def get_edge_index(
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    k: int = 5,
    dist: int = 10,
    workers: int = 1,
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
    tree = KDTree(coords_1)
    d_kdtree, idx_out = tree.query(
        coords_2, k=k, distance_upper_bound=dist, workers=workers
    )
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


def coo_to_dense_adj(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_nbrs: Optional[int] = None,
) -> torch.Tensor:

    # Check COO format
    if not edge_index.shape[0] == 2:
        msg = (
            "Edge index is not in COO format. First dimension should have "
            f"size 2, but found {edge_index.shape[0]}."
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


def predict_batch(
    lit_segger: LitSegger,
    batch: object,
    score_cut: float,
    receptive_field: dict,
    use_cc: bool = True,
) -> pd.DataFrame:
    """
    Predict cell assignments for a batch of transcript data using a segmentation model.

    Parameters
    ----------
    lit_segger : LitSegger
        The lightning module wrapping the segmentation model.
    batch : object
        A batch of transcript and cell data.
    score_cut : float
        The threshold for assigning transcripts to cells based on similarity scores.
    receptive_field : dict
        Dictionary defining the receptive field for transcript-cell and transcript-transcript relations.
    use_cc : bool, optional
        If True, perform connected components analysis for unassigned transcripts.
    knn_method : str, optional
        The method to use for nearest neighbors ('cuda' by default).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the transcript IDs, similarity scores, and assigned cell IDs.
    """

    # Get random Xenium-style ID
    def _get_id():
        id_chars = random.choices(string.ascii_lowercase, k=8)
        return "".join(id_chars) + "-nx"

    with torch.no_grad():
        batch = batch.to("cuda")

        # Assignments of cells to nuclei
        transcript_id = batch["tx"].id.cpu().numpy()
        assignments = pd.DataFrame({"transcript_id": transcript_id})

        if len(batch["bd"].id[0]) > 0:
            # Step 2.1: Calculate edge index lazily
            edge_index = get_edge_index(
                batch["bd"].pos[:, :2].cpu(),
                batch["tx"].pos[:, :2].cpu(),
                k=receptive_field["k_bd"],
                dist=receptive_field["dist_bd"],
            ).T

            # Step 2.2: Compute dense adjacency matrix
            batch["tx"]["bd_field"] = coo_to_dense_adj(
                edge_index,
                num_nodes=batch["tx"].id.shape[0],
                num_nbrs=receptive_field["k_bd"],
            )
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd")
            # 1. Get direct assignments from similarity matrix
            belongs = scores.max(1)
            assignments["score"] = belongs.values.cpu()
            mask = assignments["score"] > score_cut
            all_ids = np.concatenate(batch["bd"].id)[belongs.indices.cpu()]
            assignments.loc[mask, "segger_cell_id"] = all_ids[mask]

            if use_cc:
                # Transcript-transcript similarity scores, filtered by neighbors
                edge_index = batch["tx", "neighbors", "tx"].edge_index
                batch["tx"]["tx_field"] = coo_to_dense_adj(
                    edge_index,
                    num_nodes=batch["tx"].id.shape[0],
                )
                scores = get_similarity_scores(lit_segger.model, batch, "tx", "tx")
                scores = scores.fill_diagonal_(0)  # ignore self-similarity

                # 2. Assign remainder using connected components
                no_id = assignments["segger_cell_id"].isna().values
                no_id_scores = scores[no_id][:, no_id]
                n, comps = cc(no_id_scores, connection="weak", directed=False)
                new_ids = np.array([_get_id() for _ in range(n)])
                assignments.loc[no_id, "segger_cell_id"] = new_ids[comps]

        return assignments  # Ensure this is a pandas DataFrame


def predict(
    out_dir: os.PathLike,
    lit_segger: LitSegger,
    data_module: SeggerDataModule,
    score_cut: float,
    receptive_field: dict,
    use_cc: bool = False,
) -> pd.DataFrame:
    """
    Predict segmentation labels for multiple batches of transcript data.

    Parameters
    ----------
    out_dir : os.PathLike
        Directory to save the segmentation output file.
    lit_segger : LitSegger
        The Lightning module wrapping the segmentation model.
    data_module : SeggerDataModule
        The data module providing train/test/val data loaders.
    score_cut : float
        Similarity score threshold for transcript-to-cell assignment.
    receptive_field : dict
        Specifies transcript-cell and transcript-transcript edge parameters.
    use_cc : bool, optional
        If True, use connected components to resolve unassigned transcripts.

    Returns
    -------
    pd.DataFrame
        DataFrame containing transcript IDs, scores, and cell assignments.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        msg = f"Arg 'out_dir' must be an existing directory. Got: {out_dir}"
        raise ValueError(msg)

    # Combine assignments from training, test, and validation datasets
    predictions = []
    data_module.setup()
    for data_loader in [
        data_module.train_dataloader(),
        data_module.test_dataloader(),
        data_module.val_dataloader(),
    ]:
        for batch in data_loader:
            batch_predictions = predict_batch(
                lit_segger, batch, score_cut, receptive_field, use_cc
            )
            predictions.append(batch_predictions)
    predictions = pd.concat(predictions)

    # Keep highest cell assignment to handle overlapping tx at tile edges
    predictions = predictions.sort_values(["transcript_id", "score"])
    predictions = predictions.drop_duplicates("transcript_id", keep="last")

    # Write predictions to file
    filepath = out_dir / "segger_labeled_transcripts.parquet"
    predictions.to_parquet(filepath)
