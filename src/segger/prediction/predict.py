import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
from torch_geometric.loader import DataLoader
from torchmetrics import F1Score
from scipy.sparse.csgraph import connected_components as cc

from segger.data.utils import XeniumDataset, XeniumSample, create_anndata
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
    model = lit_segger.model
    model.eval()
    all_mappings = None
    m = torch.nn.ZeroPad2d((0, 0, 0, 1))

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to("cuda")
            y_hat = model(batch.x_dict, batch.edge_index_dict)
            y_hat["nc"] = m(y_hat["nc"])

            neighbour_idx = batch["tx"].nc_field
            similarity = torch.bmm(
                y_hat["nc"][neighbour_idx], y_hat["tx"].unsqueeze(-1)
            )
            similarity[similarity == 0] = -torch.inf
            similarity = F.sigmoid(similarity)
            adj = XeniumSample.kd_to_edge_index_(
                (len(batch["tx"].id[0]), len(batch["nc"].id[0]) + 1),
                batch["tx"].nc_field.cpu().detach().numpy(),
            ).cpu()
            adj = adj[:, adj[0, :].argsort()]
            f_output = torch.zeros(
                len(batch["tx"].id[0]), len(batch["nc"].id[0]) + 1
            )
            f_output[adj[0], adj[1]] = similarity.flatten().cpu()
            belongs = f_output.max(1)
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
            neighbour_idx = batch["tx"].tx_field
            y_tx = m(y_hat["tx"])
            similarity = torch.bmm(
                y_tx[neighbour_idx], y_hat["tx"].unsqueeze(-1)
            )
            similarity[similarity == 0] = -torch.inf
            similarity = F.sigmoid(similarity)
            adj = XeniumSample.kd_to_edge_index_(
                (len(batch["tx"].id[0]), len(batch["tx"].id[0]) + 1),
                batch["tx"].tx_field.cpu().detach().numpy(),
            ).cpu()
            adj = adj[:, adj[0, :].argsort()]
            f_output = torch.zeros(
                len(batch["tx"].id[0]), len(batch["tx"].id[0]) + 1
            )
            f_output[adj[0], adj[1]] = similarity.flatten().cpu()
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
