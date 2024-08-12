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

# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_model(checkpoint_path: str, init_emb: int, hidden_channels: int, out_channels: int, heads: int, aggr: str) -> LitSegger:
    """
    Loads the model from the checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        init_emb (int): Initial embedding size.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        heads (int): Number of attention heads.
        aggr (str): Aggregation method.

    Returns:
        LitSegger: Loaded Lightning model.
    """
    model = Segger(init_emb=init_emb, hidden_channels=hidden_channels, out_channels=out_channels, heads=heads)
    model = to_hetero(model, (['tx', 'nc'], [('tx', 'belongs', 'nc'), ('tx', 'neighbors', 'tx')]), aggr=aggr)
    litsegger = LitSegger(model)
    litsegger = litsegger.load_from_checkpoint(model, checkpoint_path=checkpoint_path, map_location=torch.device('cuda'))
    return litsegger

def predict(litsegger: LitSegger, dataset_path: str, output_path: str, score_cut: float, k_nc: int, dist_nc: int, k_tx: int, dist_tx: int) -> None:
    """
    Predicts and saves the output.

    Args:
        litsegger (LitSegger): The Lightning model.
        dataset_path (str): Path to the dataset.
        output_path (str): Path to save the output.
        score_cut (float): Score cut-off for predictions.
        k_nc (int): Number of nearest neighbors for nuclei.
        dist_nc (int): Distance threshold for nuclei.
        k_tx (int): Number of nearest neighbors for transcripts.
        dist_tx (int): Distance threshold for transcripts.
    """
    dataset = XeniumDataset(root=dataset_path)
    model = litsegger.model
    model.eval()
    all_mappings = None

    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    m = torch.nn.ZeroPad2d((0, 0, 0, 1))

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to('cuda')
            y_hat = model(batch.x_dict, batch.edge_index_dict)
            y_hat['nc'] = m(y_hat['nc'])

            neighbour_idx = batch['tx'].nc_field
            similarity = torch.bmm(y_hat['nc'][neighbour_idx], y_hat['tx'].unsqueeze(-1))
            similarity[similarity == 0] = -torch.inf
            similarity = F.sigmoid(similarity)
            adj = XeniumSample.kd_to_edge_index_((len(batch['tx'].id[0]), len(batch['nc'].id[0]) + 1), batch['tx'].nc_field.cpu().detach().numpy()).cpu()
            adj = adj[:, adj[0, :].argsort()]
            f_output = torch.zeros(len(batch['tx'].id[0]), len(batch['nc'].id[0]) + 1)
            f_output[adj[0], adj[1]] = similarity.flatten().cpu()
            belongs = f_output.max(1)
            ids_tx = np.concatenate(batch['tx'].id).reshape((1, -1))[0].astype('str')
            ids_nc = np.concatenate(batch['nc'].id).reshape((1, -1))[0].astype('str')
            belongs.indices[belongs.indices == len(ids_nc)] = 0
            mapping = np.vstack((ids_tx, ids_nc[belongs.indices], belongs.values))
            mapping[1, np.where(belongs.values < score_cut)] = 'UNASSIGNED'
            mapping[1, np.where(belongs.values == 0)] = 'FLOATING'

            idx_unknown = np.where((mapping[1,:] == 'UNASSIGNED') | (mapping[1,:] == 'FLOATING'))[0]
            ids_unknown = ids_tx[idx_unknown]
            neighbour_idx = batch['tx'].tx_field
            y_tx = m(y_hat['tx'])
            similarity = torch.bmm(y_tx[neighbour_idx], y_hat['tx'].unsqueeze(-1))
            similarity[similarity == 0] = -torch.inf
            similarity = F.sigmoid(similarity)
            adj = XeniumSample.kd_to_edge_index_((len(batch['tx'].id[0]), len(batch['tx'].id[0]) + 1), batch['tx'].tx_field.cpu().detach().numpy()).cpu()
            adj = adj[:, adj[0,:].argsort()]
            f_output = torch.zeros(len(batch['tx'].id[0]), len(batch['tx'].id[0]) + 1)
            f_output[adj[0], adj[1]] = similarity.flatten().cpu()
            f_output = f_output.fill_diagonal_(0)
            temp = torch.zeros((len(idx_unknown), len(idx_unknown))).cpu()
            temp[:len(idx_unknown), :len(idx_unknown)] = f_output[idx_unknown, :][:, idx_unknown]
            n_comps, comps = cc(temp, connection='weak', directed=False)
            random_strings = np.array([''.join(random.choices(string.ascii_lowercase, k=8)) + '-nx' for _ in range(n_comps)])
            new_ids = random_strings[comps]
            new_mapping = np.vstack((ids_unknown, new_ids, np.zeros(len(new_ids))))
            mapping = mapping[:, ~((mapping[1,:] == 'UNASSIGNED') | (mapping[1,:] == 'FLOATING'))]
            mapping = np.hstack((mapping, new_mapping))

            if all_mappings is None:
                all_mappings = mapping
            else:
                all_mappings = np.hstack((all_mappings, mapping))

    mappings_df = pd.DataFrame(all_mappings.T, columns=['transcript_id', 'segger_cell_id', 'score'])
    idx = mappings_df.groupby('transcript_id')['score'].idxmax()
    mappings_df = mappings_df.loc[idx].reset_index(drop=True)
    mappings_df.to_csv(output_path, index=False, compression='gzip')
