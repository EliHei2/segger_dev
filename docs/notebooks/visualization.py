from segger.data.parquet.sample import STSampleParquet
from segger.training.segger_data_module import SeggerDataModule
from segger.training.train import LitSegger
from segger.models.segger_model import Segger
from segger.prediction.predict import load_model
from torch_geometric.nn import to_hetero
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
import torch
import numpy as np


xenium_data_dir = Path('data_xenium')
segger_data_dir = Path('data_segger')
# Base directory to store Pytorch Lightning models
models_dir = Path('models')


sample = STSampleParquet(
    base_dir=xenium_data_dir,
    n_workers=4,
    sample_type='xenium', # this could be 'xenium_v2' in case one uses the cell boundaries from the segmentation kit.
    #weights=gene_celltype_abundance_embedding, # uncomment if gene-celltype embeddings are available
)

def extract_attention_df(attention_weights, gene_names=None, edge_type='tx-tx'):
    """
    Extract attention weights into a structured dataset.
    
    Parameters
    ----------
    attention_df : list of tuples
        List of (edge_index, alpha) tuples for each layer
    gene_names : list, optional
        List of gene names corresponding to transcript indices

    Returns
    -------
    pd.DataFrame
        DataFrame containing transcript, gene, and attention weights for each layer and head
    """
    assert edge_type in ['tx-tx', 'tx-bd'], "Edge type must be 'tx-tx' or 'tx-bd'"
    
    # Create a list to store all the data
    data = []
    
    # Process each layer
    for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
        if edge_type == 'tx-tx':
            alpha_tensor = alpha['tx']
            edge_index = edge_index['tx']
        elif edge_type == 'tx-bd':
            alpha_tensor = alpha['bd']
            edge_index = edge_index['bd']
        else:
            raise ValueError(f"Edge type must be 'tx-tx' or 'tx-bd', got {edge_type}")
        
        # Convert attention weights to numpy
        alpha_tensor = alpha_tensor.cpu().detach().numpy()
        # print(f"Alpha tensor shape: {alpha_tensor.shape}")
        edge_index = edge_index.cpu().detach().numpy()
        # print(f"Edge index shape: {edge_index.shape}")
        
        # Process each head
        for head_idx in range(alpha_tensor.shape[1]):
            # Get attention weights for this head
            head_weights = alpha_tensor[:, head_idx]
            print(f"Head {head_idx + 1} shape: {head_weights.shape}")
            print(f"Edge index shape: {edge_index.shape}")
            
            # Create entries for each edge
            for i, (src, dst) in enumerate(edge_index.T):
                entry = {
                    'source': int(src),
                    'target': int(dst),
                    'edge_type': edge_type,
                    'layer': layer_idx + 1,
                    'head': head_idx + 1,
                    'attention_weight': float(head_weights[i])
                }
                
                # Add gene names if available
                if gene_names is not None:
                    entry['source_gene'] = gene_names[src]
                    entry['target_gene'] = gene_names[dst]
                
                data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def visualize_attention_df(attention_df, layer_idx, head_idx, edge_type, gene_names=None):
    """
    Visualize attention weights of a data frame as a heatmap.
    
    Parameters
    ----------
    attention_df : torch.Tensor
        Attention weights of a data frame. Keys: 'source', 'target', 'edge_type', 'layer', 'head', 'attention_weight'.
    layer_idx : int
        Layer index.
    head_idx : int
        Head index.
    edge_type : str
        Edge type.
    gene_names : list
        a list of gene names ordered by the transcript indices.
        
    """
    # Get number of nodes
    num_nodes = len(attention_df['source'].unique())
    # Extract attention weights with given layer_idx and head_idx
    attention_df = attention_df[attention_df['layer'] == layer_idx + 1]
    attention_df = attention_df[attention_df['head'] == head_idx + 1]
    
    # Create adjacency matrix for visualization
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # Fill adjacency matrix with attention weights
    for _, row in attention_df.iterrows():
        src = row['source']
        dst = row['target']
        adj_matrix[src, dst] = row['attention_weight']
    
    # Sort by genes if gene names are provided
    if gene_names is not None:
        # Sort nodes by gene names
        sorted_indices = sorted(attention_df['source'].unique(), key=lambda i: gene_names[i])
        # Reorder the adjacency matrix
        adj_matrix = adj_matrix[sorted_indices][:, sorted_indices]
        # Get sorted gene names for labels
        sorted_genes = [gene_names[i] for i in sorted_indices]
    else:
        sorted_genes = None
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap='viridis', annot=False, cbar=True)
    plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}')
    plt.xlabel('Target Node (Transcript)')
    plt.ylabel('Source Node (Transcript)')
    
    # Add gene labels if available
    if sorted_genes is not None:
        # Add gene labels to the plot
        plt.xticks(np.arange(len(sorted_genes)) + 0.5, sorted_genes, rotation=90, ha='right')
        plt.yticks(np.arange(len(sorted_genes)) + 0.5, sorted_genes, rotation=0)
    
    plt.tight_layout()
    plt.savefig(Path('figures') / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png')
    plt.close()
    
# Main function to load model and visualize attention weights
def main():
    # Paths to data and models
    model_version = 1
    model_path = Path('models') / "lightning_logs" / f"version_{model_version}"
    ls = load_model(model_path / "checkpoints")

    ls.eval()

    # load transcripts
    transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')

    # Move batch to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ls = ls.to(device)

    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=Path('data_segger'),
        batch_size=2,
        num_workers=2,
    )

    dm.setup()

    # Get gene names from the batch data and transcripts
    # Convert tensor IDs to numpy array and then to list for indexing
    transcript_ids = batch['tx'].id.cpu().numpy()
    # Create a mapping from transcript ID to gene name
    id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    # Get gene names in the same order as the batch
    gene_names = [id_to_gene[id] for id in transcript_ids]

    # Get a sample batch from the data module
    batch = dm.train[0].to(device)

    # Run forward pass to get attention weights
    with torch.no_grad():
        # Access the heterogeneous model
        hetero_model = ls.model
        # Get node features and edge indices
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        
        # Run forward pass through the model
        _, attention_weights = hetero_model(x_dict, edge_index_dict)

    edge_type = "tx-tx"

    # Extract attention weights into a structured dataset
    attention_df = extract_attention_df(attention_weights, gene_names)

    # Save the attention weights dataset
    output_path = Path(f'figures/attention_weights_{edge_type}.csv')
    attention_df.to_csv(output_path, index=False)
    print(f"Saved attention weights dataset to {output_path}")

    # visualize attention weights
    num_nodes = batch.x_dict['tx'].shape[0]
        
    for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
        print(f"Layer {layer_idx + 1}: edge_index type = {type(edge_index)}, edge keys: {list(edge_index.keys())}, alpha type = {type(alpha)}, alpha keys: {list(alpha.keys())}")
        
        if edge_type == 'tx-tx':
            alpha_tensor = alpha['tx']
            edge_index_tensor = edge_index['tx']
        elif edge_type == 'tx-bd':
            alpha_tensor = alpha['bd']
            edge_index_tensor = edge_index['bd']
        else:
            raise ValueError(f"Edge type must be 'tx-tx' or 'tx-bd', got {edge_type}")
        
        print(f"Alpha tensor shape: {alpha_tensor.shape}")
            
        for head_idx in range(alpha_tensor.shape[1]):
            visualize_attention_df(
                attention_df=attention_df,
                layer_idx=layer_idx,
                head_idx=head_idx,
                edge_type=edge_type,
                gene_names=gene_names
            )

if __name__ == '__main__':
    main()