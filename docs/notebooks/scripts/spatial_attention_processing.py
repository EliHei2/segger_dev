import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from typing import Dict
from torch_geometric.data import HeteroData
import pickle

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from docs.notebooks.visualization.batch_visualization import extract_attention_df
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import load_model

def process_attention_weights(attention_df: pd.DataFrame, batch: HeteroData) -> Dict[str, torch.Tensor]:
    attention_dict = {}
    for edge_type in attention_df['edge_type'].unique():
        edge_df = attention_df[attention_df['edge_type'] == edge_type]
        if edge_type == "tx-tx":
            edge_index = batch[("tx", "neighbors", "tx")].edge_index.cpu().numpy()
        else:
            edge_index = batch[("tx", "belongs", "bd")].edge_index.cpu().numpy()
        weight_map = {}
        for _, row in edge_df.iterrows():
            key = (row['source'], row['target'])
            if key not in weight_map:
                weight_map[key] = 0.0
            weight_map[key] += row['attention_weight']
        attn_weights = np.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            key = (source.item(), target.item())
            if key in weight_map:
                attn_weights[i] = weight_map[key]
        attention_dict[edge_type] = torch.tensor(attn_weights)
    return attention_dict

def main():
    transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')
    idx_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    gene_names = transcripts['feature_name'].unique()
    cell_types = pd.read_csv(Path('data_xenium') / 'cell_groups.csv')
    cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
    cell_types_order = pd.read_csv(Path('data_xenium') / 'cell_types_order_color.csv')
    cell_type_to_color = dict(zip(cell_types_order['Cell Type'], cell_types_order['Color']))
    gene_types = pd.read_csv(Path('data_xenium') / 'gene_groups_ordered_color.csv')
    gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
    gene_type_to_color = dict(zip(gene_types['group'], gene_types['Color']))
    if not (Path('intermediate_data') / f'processed_attention.pkl').exists() or not (Path('intermediate_data') / f'batch_0.pkl').exists():
        model_version = 1
        model_path = Path('models') / "lightning_logs" / f"version_{model_version}"
        ls = load_model(model_path / "checkpoints")
        ls.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ls = ls.to(device)
        dm = SeggerDataModule(
            data_dir=Path('data_segger'),
            batch_size=2,
            num_workers=2,
        )
        dm.setup()
        batch = dm.train[0].to(device)
        cell_id = batch['bd'].id
        idx_to_cell = {i: cell_id[i] for i in range(len(cell_id))}
        transcript_ids = batch['tx'].id.cpu().numpy()
        gene_names_batch = [idx_to_gene[tx_id] for tx_id in transcript_ids]
        cell_id_batch = batch['bd'].id
        with torch.no_grad():
            hetero_model = ls.model
            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            _, attention_weights = hetero_model(x_dict, edge_index_dict)
        attention_bd = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_id_batch, edge_type = "tx-bd", cell_types_dict = cell_types_dict)
        attention_tx = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_id_batch, edge_type = "tx-tx", cell_types_dict = cell_types_dict)
        attention_df = pd.concat([attention_bd, attention_tx], ignore_index=True)
        processed_attention = process_attention_weights(attention_df, batch)
        with open(Path('intermediate_data') / f'processed_attention.pkl', 'wb') as f:
            pickle.dump(processed_attention, f)
        with open(Path('intermediate_data') / f'batch_0.pkl', 'wb') as f:
            pickle.dump(batch, f)
        print("Processed attention and batch saved to intermediate_data/.")
    else:
        print("Processed attention and batch already exist in intermediate_data/.")

if __name__ == "__main__":
    main()