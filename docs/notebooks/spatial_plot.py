import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
from pathlib import Path
import numpy as np
from docs.notebooks.visualization.spatial_visualization import visualize_tile_with_attention_interactive
from docs.notebooks.visualization.batch_visualization import extract_attention_df
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import load_model
import torch
from typing import Dict
from torch_geometric.data import HeteroData
import pickle

def process_attention_weights(attention_df: pd.DataFrame, batch: HeteroData) -> Dict[str, torch.Tensor]:
    """
    Process attention weights DataFrame to create a dictionary of edge indices and attention weights.
    
    Parameters
    ----------
    attention_df : pd.DataFrame
        DataFrame containing attention weights with columns: source, target, layer, edge_type, head, attention_weight
    batch : HeteroData
        PyTorch Geometric HeteroData object containing the batch data
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing attention weights for each edge type
    """
    # Initialize dictionary to store attention weights
    attention_dict = {}
    
    # Process each edge type
    for edge_type in attention_df['edge_type'].unique():
        # Filter attention weights for this edge type
        edge_df = attention_df[attention_df['edge_type'] == edge_type]
        
        # Get edge indices from the batch
        if edge_type == "tx-tx":
            edge_index = batch[("tx", "neighbors", "tx")].edge_index.cpu().numpy()
        else:  # tx-bd
            edge_index = batch[("tx", "belongs", "bd")].edge_index.cpu().numpy()
        
        # Create a mapping from (source, target) to attention weight
        weight_map = dict(zip(zip(edge_df['source'], edge_df['target']), edge_df['attention_weight']))
        
        # Create attention weights array matching edge_index shape
        attn_weights = np.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            key = (source.item(), target.item())
            if key in weight_map:
                attn_weights[i] = weight_map[key]
        
        # Convert to tensor and store in dictionary
        attention_dict[edge_type] = torch.tensor(attn_weights)
    
    return attention_dict

def main():
    # Load transcripts
    transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')
    # Get id of transcripts to gene names
    idx_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))

    # Get cell types
    cell_types = pd.read_csv(Path('data_xenium') / 'cell_groups.csv')
    cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
    cell_types_order = pd.read_csv(Path('data_xenium') / 'cell_types_order_color.csv')
    cell_type_to_color = dict(zip(cell_types_order['Cell Type'], cell_types_order['Color']))
    
    # Get gene types
    gene_types = pd.read_csv(Path('data_xenium') / 'gene_groups_ordered_color.csv')
    gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
    gene_type_to_color = dict(zip(gene_types['group'], gene_types['Color']))

    if not (Path('intermediate_data') / f'processed_attention.pkl').exists() or not (Path('intermediate_data') / f'batch_0.pkl').exists():
        # Paths to data and models
        model_version = 1
        model_path = Path('models') / "lightning_logs" / f"version_{model_version}"
        ls = load_model(model_path / "checkpoints")
        ls.eval()

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ls = ls.to(device)

        # Initialize the Lightning data module
        dm = SeggerDataModule(
            data_dir=Path('data_segger'),
            batch_size=2,
            num_workers=2,
        )
        dm.setup()

        # Get a sample batch
        batch = dm.train[0].to(device)

        # get the cell id of the batch
        cell_id = batch['bd'].id
        idx_to_cell = {i: cell_id[i] for i in range(len(cell_id))}

        # Get gene names
        transcript_ids = batch['tx'].id.cpu().numpy()
        gene_names_batch = [idx_to_gene[tx_id] for tx_id in transcript_ids]

        # Get cell ids
        cell_id_batch = batch['bd'].id

        # Run forward pass to get attention weights
        with torch.no_grad():
            hetero_model = ls.model
            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            _, attention_weights = hetero_model(x_dict, edge_index_dict)

        # Extract attention weights for both edge types
        attention_bd = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_id_batch, edge_type = "tx-bd", cell_types_dict = cell_types_dict)
        attention_tx = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_id_batch, edge_type = "tx-tx", cell_types_dict = cell_types_dict)

        # Combine attention weights
        attention_df = pd.concat([attention_bd, attention_tx], ignore_index=True)

        # Process attention weights to match the format needed by visualization
        processed_attention = process_attention_weights(attention_df, batch)

        # save processed attention
        with open(Path('intermediate_data') / f'processed_attention.pkl', 'wb') as f:
            pickle.dump(processed_attention, f)
            
        # save batch
        with open(Path('intermediate_data') / f'batch_0.pkl', 'wb') as f:
            pickle.dump(batch, f)
    else:
        with open(Path('intermediate_data') / f'processed_attention.pkl', 'rb') as f:
            processed_attention = pickle.load(f)
        with open(Path('intermediate_data') / f'batch_0.pkl', 'rb') as f:
            batch = pickle.load(f)
            
        # get the cell id of the batch
        cell_id = batch['bd'].id
        idx_to_cell = {i: cell_id[i] for i in range(len(cell_id))}
    
    # # filter out transcripts with gene_name in gene_restriction
    # top_k = 50 # max: 50
    # with open(Path('intermediate_data') / f'top_genes_k50.pkl', 'rb') as f:
    #     top_genes, _ = pickle.load(f)
    
    # top_genes_restricted = top_genes[::-1][:top_k]
    # print(f"Top {top_k} genes: {top_genes_restricted}")

    visualize_tile_with_attention_interactive(
        batch=batch,
        attention_weights=processed_attention,
        idx_to_gene=idx_to_gene,
        idx_to_cell=idx_to_cell,
        gene_types_dict=gene_types_dict,
        cell_types_dict=cell_types_dict,
        cell_type_to_color=cell_type_to_color,
        gene_type_to_color=gene_type_to_color,
        boundaries_file=Path('data_xenium') / 'cell_boundaries.parquet',
        nucleus_file=Path('data_xenium') / 'nucleus_boundaries.parquet'
    )
    
if __name__ == "__main__":
    main()