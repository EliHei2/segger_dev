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
import pickle
from scipy.sparse import csr_matrix, lil_matrix
import os
import time
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
            # print(f"Head {head_idx + 1} shape: {head_weights.shape}")
            # print(f"Edge index shape: {edge_index.shape}")
            
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
    # if sorted_genes is not None:
    #     # Add gene labels to the plot
    #     plt.xticks(np.arange(len(sorted_genes)) + 0.5, sorted_genes, rotation=90, ha='right')
    #     plt.yticks(np.arange(len(sorted_genes)) + 0.5, sorted_genes, rotation=0)
    
    plt.tight_layout()
    plt.savefig(Path('figures') / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png')
    plt.close()


def visualize_attention_difference(attention_df, edge_type, compare_type='layers', layer_indices=None, head_indices=None, gene_names=None, max_matrix_size=1000, downsample_threshold=500):
    """
    Visualize differences in attention weights across layers or heads at transcript level.
    
    Parameters
    ----------
    attention_df : pd.DataFrame
        Attention weights dataframe with columns: 'source', 'target', 'edge_type', 'layer', 'head', 'attention_weight'.
    edge_type : str
        Edge type to filter by (e.g., 'tx-tx', 'tx-bd').
    compare_type : str
        Type of comparison: 'layers' or 'heads'.
    layer_indices : list
        List of layer indices to compare. If None, uses all available layers.
    head_indices : list
        List of head indices to compare. If None, uses all available heads.
    gene_names : list
        List of gene names ordered by the transcript indices.
    max_matrix_size : int
        Maximum size of matrix (n×n) to process. If larger, it will be downsampled.
    downsample_threshold : int
        Threshold for matrix size beyond which to apply downsampling.
    """
    
    # Filter by edge_type
    filtered_df = attention_df[attention_df['edge_type'] == edge_type]
    
    # Get all available layers and heads if not specified
    if layer_indices is None:
        layer_indices = sorted(filtered_df['layer'].unique())
    if head_indices is None:
        head_indices = sorted(filtered_df['head'].unique())
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True, exist_ok=True)
    
    total_comparisons = 0
    completed_comparisons = 0
    
    # Helper function to downsample a matrix
    def downsample_matrix(matrix, target_size):
        """Downsample a matrix to the target size using averaging."""
        h, w = matrix.shape
        if max(h, w) <= target_size:
            return matrix
        
        # Calculate the downsampling factor
        factor = max(1, int(max(h, w) / target_size))
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        
        # Use block_reduce for efficient downsampling
        from skimage.measure import block_reduce
        return block_reduce(matrix, block_size=(factor, factor), func=np.mean)
    
    if compare_type == 'layers':
        # Calculate total number of comparisons
        for head_idx in head_indices:
            head_df = filtered_df[filtered_df['head'] == head_idx]
            if not head_df.empty and len(layer_indices) >= 2:
                total_comparisons += len(layer_indices) * (len(layer_indices) - 1) // 2
        
        # Compare differences between layers for each head
        for h_idx, head_idx in enumerate(head_indices):
            head_df = filtered_df[filtered_df['head'] == head_idx]
            
            # Skip if no data for this head
            if head_df.empty:
                continue
                
            # We need at least 2 layers to compare
            if len(layer_indices) < 2:
                continue
            
            # Get number of nodes
            num_nodes = len(head_df['source'].unique())
            if num_nodes == 0:
                continue
            
            # Check if matrix is too large
            if num_nodes > max_matrix_size:
                # Randomly sample nodes to reduce matrix size
                all_nodes = sorted(head_df['source'].unique())
                np.random.seed(42)  # For reproducibility
                sampled_nodes = np.random.choice(all_nodes, size=max_matrix_size, replace=False)
                head_df = head_df[head_df['source'].isin(sampled_nodes) & head_df['target'].isin(sampled_nodes)]
                num_nodes = len(sampled_nodes)
            
            # For each pair of layers, compute difference
            for i in range(len(layer_indices) - 1):
                for j in range(i + 1, len(layer_indices)):
                    layer1 = layer_indices[i]
                    layer2 = layer_indices[j]
                    
                    # Create adjacency matrices for both layers with numpy arrays directly
                    # (more memory efficient for large matrices than lil_matrix conversion)
                    adj_matrix1 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                    adj_matrix2 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                    
                    # Fill adjacency matrices
                    layer1_df = head_df[head_df['layer'] == layer1]
                    layer2_df = head_df[head_df['layer'] == layer2]
                    
                    # Skip if either layer has no data
                    if layer1_df.empty or layer2_df.empty:
                        continue
                    
                    # Map node indices to 0-based indices for more efficient matrix construction
                    unique_nodes = sorted(set(head_df['source'].unique()) | set(head_df['target'].unique()))
                    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
                    
                    # Fill matrices using mapped indices
                    for _, row in layer1_df.iterrows():
                        src = node_to_idx[row['source']]
                        dst = node_to_idx[row['target']]
                        adj_matrix1[src, dst] += row['attention_weight']
                    
                    for _, row in layer2_df.iterrows():
                        src = node_to_idx[row['source']]
                        dst = node_to_idx[row['target']]
                        adj_matrix2[src, dst] += row['attention_weight']
                    
                    # Compute difference
                    diff_matrix = adj_matrix2 - adj_matrix1
                    
                    # Sort by genes if gene names are provided
                    if gene_names is not None:
                        # Create a mapping from original node indices to gene names
                        node_genes = {}
                        for node in unique_nodes:
                            if node < len(gene_names):
                                node_genes[node] = gene_names[node]
                        
                        if not node_genes:
                            # Continue without sorting
                            continue
                        else:
                            # Map index-to-index for sorting based on gene names
                            sorted_idx_map = sorted(range(len(unique_nodes)), 
                                                key=lambda i: node_genes.get(unique_nodes[i], "") if unique_nodes[i] in node_genes else "")
                            
                            # Reorder the difference matrix
                            diff_matrix = diff_matrix[sorted_idx_map][:, sorted_idx_map]
                    
                    # Check if matrix needs downsampling for plotting
                    if max(diff_matrix.shape) > downsample_threshold:
                        diff_matrix = downsample_matrix(diff_matrix, downsample_threshold)
                    
                    # Plot heatmap of differences
                    plt.figure(figsize=(12, 10))
                    
                    # Use diverging colormap for difference visualization
                    vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
                    if vmax == 0:
                        vmax = 1  # Avoid division by zero
                    
                    # Use a different backend for large matrices to improve performance
                    if max(diff_matrix.shape) > 200:
                        plt.ioff()  # Turn off interactive mode for faster plotting
                    try:
                        # Use imshow instead of heatmap for better performance with large matrices
                        if max(diff_matrix.shape) > 200:
                            plt.imshow(diff_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax)
                            plt.colorbar()
                        else:
                            sns.heatmap(diff_matrix, cmap='coolwarm', annot=False, cbar=True, vmin=-vmax, vmax=vmax)
                        
                        plt.title(f'Attention Weight Difference - Layer {layer2} vs Layer {layer1}, Head {head_idx}, Edge Type: {edge_type}')
                        plt.xlabel('Target Node (Transcript)')
                        plt.ylabel('Source Node (Transcript)')
                        
                        plt.tight_layout()
                        save_path = figures_dir / f'attention_diff_layer{layer2}vs{layer1}_head{head_idx}_{edge_type}.png'
                        plt.savefig(save_path, dpi=150)
                        plt.close('all')  # Close all figures to free memory
                    except Exception as e:
                        plt.close('all')  # Make sure to close any open figures
                    
                    # Clear memory
                    del adj_matrix1, adj_matrix2, diff_matrix
                    
                    completed_comparisons += 1
    
    elif compare_type == 'heads':
        # Calculate total number of comparisons
        for layer_idx in layer_indices:
            layer_df = filtered_df[filtered_df['layer'] == layer_idx]
            if not layer_df.empty and len(head_indices) >= 2:
                total_comparisons += len(head_indices) * (len(head_indices) - 1) // 2
        
        
        # Compare differences between heads for each layer
        for l_idx, layer_idx in enumerate(layer_indices):
            layer_time = time.time()
            
            layer_df = filtered_df[filtered_df['layer'] == layer_idx]
            
            # Skip if no data for this layer
            if layer_df.empty:
                continue
                
            # We need at least 2 heads to compare
            if len(head_indices) < 2:
                continue
            
            # Get number of nodes
            num_nodes = len(layer_df['source'].unique())
            if num_nodes == 0:
                continue
                
            
            # Check if matrix is too large
            if num_nodes > max_matrix_size:
                # Randomly sample nodes to reduce matrix size
                all_nodes = sorted(layer_df['source'].unique())
                np.random.seed(42)  # For reproducibility
                sampled_nodes = np.random.choice(all_nodes, size=max_matrix_size, replace=False)
                layer_df = layer_df[layer_df['source'].isin(sampled_nodes) & layer_df['target'].isin(sampled_nodes)]
                num_nodes = len(sampled_nodes)
            
            # For each pair of heads, compute difference
            for i in range(len(head_indices) - 1):
                for j in range(i + 1, len(head_indices)):
                    head1 = head_indices[i]
                    head2 = head_indices[j]
                    
                    # Create adjacency matrices for both heads with numpy arrays directly
                    adj_matrix1 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                    adj_matrix2 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                    
                    # Fill adjacency matrices
                    head1_df = layer_df[layer_df['head'] == head1]
                    head2_df = layer_df[layer_df['head'] == head2]
                    
                    # Skip if either head has no data
                    if head1_df.empty or head2_df.empty:
                        continue
                    
                    
                    # Map node indices to 0-based indices for more efficient matrix construction
                    unique_nodes = sorted(set(layer_df['source'].unique()) | set(layer_df['target'].unique()))
                    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
                    
                    # Fill matrices using mapped indices
                    for _, row in head1_df.iterrows():
                        src = node_to_idx[row['source']]
                        dst = node_to_idx[row['target']]
                        adj_matrix1[src, dst] += row['attention_weight']
                    
                    for _, row in head2_df.iterrows():
                        src = node_to_idx[row['source']]
                        dst = node_to_idx[row['target']]
                        adj_matrix2[src, dst] += row['attention_weight']
                    
                    # Compute difference
                    diff_matrix = adj_matrix2 - adj_matrix1
                    
                    # Sort by genes if gene names are provided
                    if gene_names is not None:
                        sort_time = time.time()
                        # Create a mapping from original node indices to gene names
                        node_genes = {}
                        for node in unique_nodes:
                            if node < len(gene_names):
                                node_genes[node] = gene_names[node]
                        
                        if not node_genes:
                            # Continue without sorting
                            continue
                        else:
                            # Map index-to-index for sorting based on gene names
                            sorted_idx_map = sorted(range(len(unique_nodes)), 
                                                key=lambda i: node_genes.get(unique_nodes[i], "") if unique_nodes[i] in node_genes else "")
                            
                            # Reorder the difference matrix
                            diff_matrix = diff_matrix[sorted_idx_map][:, sorted_idx_map]
                    
                    # Check if matrix needs downsampling for plotting
                    if max(diff_matrix.shape) > downsample_threshold:
                        diff_matrix = downsample_matrix(diff_matrix, downsample_threshold)
                    
                    # Plot heatmap of differences
                    plt.figure(figsize=(12, 10))
                    
                    # Use diverging colormap for difference visualization
                    vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
                    if vmax == 0:
                        vmax = 1  # Avoid division by zero
                    
                    
                    # Use a different backend for large matrices to improve performance
                    if max(diff_matrix.shape) > 200:
                        plt.ioff()  # Turn off interactive mode for faster plotting
                        
                    try:
                        # Use imshow instead of heatmap for better performance with large matrices
                        if max(diff_matrix.shape) > 200:
                            plt.imshow(diff_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax)
                            plt.colorbar()
                        else:
                            sns.heatmap(diff_matrix, cmap='coolwarm', annot=False, cbar=True, vmin=-vmax, vmax=vmax)
                        
                        plt.title(f'Attention Weight Difference - Head {head2} vs Head {head1}, Layer {layer_idx}, Edge Type: {edge_type}')
                        plt.xlabel('Target Node (Transcript)')
                        plt.ylabel('Source Node (Transcript)')
                        
                        plt.tight_layout()
                        save_path = figures_dir / f'attention_diff_head{head2}vs{head1}_layer{layer_idx}_{edge_type}.png'
                        plt.savefig(save_path, dpi=150)
                        plt.close('all')  # Close all figures to free memory
                    except Exception as e:
                        plt.close('all')  # Make sure to close any open figures
                    
                    # Clear memory
                    del adj_matrix1, adj_matrix2, diff_matrix
                    
                    completed_comparisons += 1
    else:
        raise ValueError(f"Invalid comparison type: {compare_type}. Must be 'layers' or 'heads'.")

def summarize_attention_by_gene_df(attention_df, layer_idx, head_idx, edge_type, gene_to_idx, visualize=True):
    """
    Visualize attention weights of a data frame as a heatmap.
    
    Parameters
    ----------
    attention_df : pd.DataFrame
        Attention weights dataframe with columns: 'source', 'target', 'edge_type', 'layer', 'head', 'attention_weight', 'source_gene', 'target_gene'.
    layer_idx : int
        Layer index.
    head_idx : int
        Head index.
    edge_type : str
        Edge type.
    gene_to_idx : dict
        a dictionary of gene names to indices.
    """
    # Extract attention weights with given layer_idx and head_idx
    attention_df = attention_df[attention_df['layer'] == layer_idx + 1]
    attention_df = attention_df[attention_df['head'] == head_idx + 1]
    
    # Get unique genes by gene_to_idx
    num_genes = len(gene_to_idx)
    
    # Create adjacency matrix for visualization - use lil_matrix for efficient construction
    adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
    count_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
    
    # Fill adjacency matrix with attention weights
    for _, row in attention_df.iterrows():
        src_idx = gene_to_idx[row['source_gene']]
        dst_idx = gene_to_idx[row['target_gene']]
        adj_matrix[src_idx, dst_idx] += row['attention_weight']
        count_matrix[src_idx, dst_idx] += 1
    
    # Convert to CSR format for efficient computation
    adj_matrix = adj_matrix.tocsr()
    count_matrix = count_matrix.tocsr()
    
    if visualize:
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix.toarray()/count_matrix.toarray(), cmap='viridis', annot=False, cbar=True)
        plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}')
        plt.xlabel('Target Gene')
        plt.ylabel('Source Gene')
        
        # add gene labels
        # plt.xticks(np.arange(len(gene_to_idx)) + 0.5, sorted(gene_to_idx.keys()), rotation=90, ha='right')
        # plt.yticks(np.arange(len(gene_to_idx)) + 0.5, sorted(gene_to_idx.keys()), rotation=0)
        
        plt.tight_layout()
        plt.savefig(Path('figures') / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png')
        plt.close()
    return adj_matrix, count_matrix

def safe_divide_sparse_numpy(A, B):
    # Convert to COO format to ensure indices match
    A_coo = A.tocoo()
    B_coo = B.tocoo()
    
    # Create result matrix with same structure as A
    result = A.copy()
    
    # Only divide where B is non-zero
    mask = B_coo.data > 0
    result.data[mask] = A_coo.data[mask] / B_coo.data[mask]
    # Where B is zero, set result to zero (or another default)
    result.data[~mask] = 0
    
    return result

def compare_attention_patterns(attention_gene_matrix_dict, comparison_type='layers', layer_indices=None, head_indices=None, edge_type='tx-tx', top_k=20, gene_to_idx=None):
    """
    Create comparison plots between different attention layers or heads.
    
    Parameters
    ----------
    attention_gene_matrix_dict : dict
        Dictionary containing 'adj_matrix' and 'count_matrix'.
    comparison_type : str
        Type of comparison: 'layers' or 'heads'.
    layer_indices : list
        List of layer indices to compare. If None, uses all layers.
    head_indices : list
        List of head indices to compare. If None, uses all heads.
    edge_type : str
        Edge type to visualize.
    top_k : int
        Number of top attention pairs to visualize.
    gene_to_idx : dict
        Dictionary mapping gene names to indices.
    """
    # Create output directory
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    layers = len(attention_gene_matrix_dict['adj_matrix'])
    heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Set default indices if not provided
    if layer_indices is None:
        layer_indices = list(range(layers))
    if head_indices is None:
        head_indices = list(range(heads))
    
    # Get the reverse mapping: idx to gene
    if gene_to_idx:
        idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    
    if comparison_type == 'layers':
        # Compare across layers for each head
        for head_idx in head_indices:
            # Create a figure for this head
            fig, axes = plt.subplots(1, len(layer_indices), figsize=(6*len(layer_indices), 5))
            if len(layer_indices) == 1:
                axes = [axes]  # Make it iterable
                
            # Get matrices for each layer with this head
            for i, layer_idx in enumerate(layer_indices):
                matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
                
                # Plot heatmap
                im = sns.heatmap(matrix, cmap='viridis', ax=axes[i], cbar=(i == len(layer_indices)-1))
                axes[i].set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
                axes[i].set_xlabel('Target Gene')
                axes[i].set_ylabel('Source Gene')
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'layer_comparison_head_{head_idx+1}_{edge_type}.png')
            plt.close()
            
            # Create a plot showing top attention pairs across layers
            plt.figure(figsize=(12, 8))
            
            # Collect top gene pairs from each layer
            all_pairs = []
            for layer_idx in layer_indices:
                matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
                # Flatten and find top_k indices
                flat_indices = np.argsort(matrix.flatten())[-top_k:]
                # Convert flat indices to 2D indices
                for flat_idx in flat_indices:
                    src_idx = flat_idx // matrix.shape[1]
                    dst_idx = flat_idx % matrix.shape[1]
                    if gene_to_idx:
                        pair_name = f"{idx_to_gene[src_idx]}->{idx_to_gene[dst_idx]}"
                    else:
                        pair_name = f"Gene {src_idx}->Gene {dst_idx}"
                    all_pairs.append((pair_name, layer_idx, matrix[src_idx, dst_idx]))
            
            # Create DataFrame for easy plotting
            df = pd.DataFrame(all_pairs, columns=['Gene Pair', 'Layer', 'Attention Weight'])
            # Sort by attention weight for better visualization
            df = df.sort_values('Attention Weight', ascending=False).head(top_k)
            
            # Plot as grouped bar chart
            sns.barplot(data=df, x='Gene Pair', y='Attention Weight', hue='Layer', palette='viridis')
            plt.title(f'Top {top_k} Attention Weights Across Layers (Head {head_idx+1})')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(figures_dir / f'top_attention_across_layers_head_{head_idx+1}_{edge_type}.png')
            plt.close()
            
    elif comparison_type == 'heads':
        # Compare across heads for each layer
        for layer_idx in layer_indices:
            # Create a figure for this layer
            fig, axes = plt.subplots(1, len(head_indices), figsize=(6*len(head_indices), 5))
            if len(head_indices) == 1:
                axes = [axes]  # Make it iterable
                
            # Get matrices for each head with this layer
            for i, head_idx in enumerate(head_indices):
                matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
                
                # Plot heatmap
                im = sns.heatmap(matrix, cmap='viridis', ax=axes[i], cbar=(i == len(head_indices)-1))
                axes[i].set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
                axes[i].set_xlabel('Target Gene')
                axes[i].set_ylabel('Source Gene')
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'head_comparison_layer_{layer_idx+1}_{edge_type}.png')
            plt.close()
            
            # Create a plot showing top attention pairs across heads
            plt.figure(figsize=(12, 8))
            
            # Collect top gene pairs from each head
            all_pairs = []
            for head_idx in head_indices:
                matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
                # Flatten and find top_k indices
                flat_indices = np.argsort(matrix.flatten())[-top_k:]
                # Convert flat indices to 2D indices
                for flat_idx in flat_indices:
                    src_idx = flat_idx // matrix.shape[1]
                    dst_idx = flat_idx % matrix.shape[1]
                    if gene_to_idx:
                        pair_name = f"{idx_to_gene[src_idx]}->{idx_to_gene[dst_idx]}"
                    else:
                        pair_name = f"Gene {src_idx}->Gene {dst_idx}"
                    all_pairs.append((pair_name, head_idx, matrix[src_idx, dst_idx]))
            
            # Create DataFrame for easy plotting
            df = pd.DataFrame(all_pairs, columns=['Gene Pair', 'Head', 'Attention Weight'])
            # Sort by attention weight for better visualization
            df = df.sort_values('Attention Weight', ascending=False).head(top_k)
            
            # Plot as grouped bar chart
            sns.barplot(data=df, x='Gene Pair', y='Attention Weight', hue='Head', palette='viridis')
            plt.title(f'Top {top_k} Attention Weights Across Heads (Layer {layer_idx+1})')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(figures_dir / f'top_attention_across_heads_layer_{layer_idx+1}_{edge_type}.png')
            plt.close()
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}. Must be 'layers' or 'heads'.")

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

    # --------------------------------------------------------------------
    # ---------------------- transcript attention weights ----------------------
    # --------------------------------------------------------------------
    
    # Get a sample batch from the data module
    batch = dm.train[0].to(device)
    
    # Get gene names from the batch data and transcripts
    # Convert tensor IDs to numpy array and then to list for indexing
    transcript_ids = batch['tx'].id.cpu().numpy()
    # Create a mapping from transcript ID to gene name
    id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    # Get gene names in the same order as the batch
    gene_names = [id_to_gene[id] for id in transcript_ids]

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

    # # Save the attention weights dataset
    # output_path = Path(f'figures/attention_weights_{edge_type}.csv')
    # attention_df.to_csv(output_path, index=False)
    # print(f"Saved attention weights dataset to {output_path}")
        
    # Get the first layer's attention weights
    # layers = 2
    # heads = 2
    # for layer_idx in range(layers):
    #     for head_idx in range(heads):
    #         visualize_attention_df(
    #             attention_df=attention_df,
    #             layer_idx=layer_idx,
    #             head_idx=head_idx,
    #             edge_type=edge_type,
    #             gene_names=gene_names
    #         )

    # Visualize transcript-level attention differences
    print("Visualizing transcript-level attention differences...")
    # Extract attention weights into a structured dataset for the whole batch
    attention_df = extract_attention_df(attention_weights, gene_names)
    
    # Set layer and head indices for the comparison
    layer_indices = [1, 2]  # Assuming 1-indexed layers in the dataframe
    head_indices = [1, 2]   # Assuming 1-indexed heads in the dataframe
    
    # Visualize differences between layers for each head
    visualize_attention_difference(
        attention_df=attention_df,
        edge_type=edge_type,
        compare_type='layers',
        layer_indices=layer_indices,
        head_indices=head_indices,
        gene_names=gene_names
    )
    
    # Visualize differences between heads for each layer
    visualize_attention_difference(
        attention_df=attention_df,
        edge_type=edge_type,
        compare_type='heads',
        layer_indices=layer_indices,
        head_indices=head_indices,
        gene_names=gene_names
    )
    
    # --------------------------------------------------------------------
    # ---------------------- gene attention weights ----------------------
    # --------------------------------------------------------------------
    # # create a matrix of gene attention weights
    # num_genes = len(transcripts['feature_name'].unique()) # 538
    # # construct the gene to index mapping
    # gene_to_idx = {gene: idx for idx, gene in enumerate(transcripts['feature_name'].unique())}
    # # Create a mapping from transcript ID to gene name
    # id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))

    # layers = 2
    # heads = 2

    # attention_gene_matrix_dict = {
    #     "adj_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32).tocsr() for _ in range(heads)] for _ in range(layers)],
    #     "count_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32).tocsr() for _ in range(heads)] for _ in range(layers)]
    # }

    # count = 0
    # time_start = time.time()
    # # travel through each batch and add up the adj_matrix and count_matrix
    # for batch in dm.train[0:1]:
    #     with torch.no_grad():
    #         count += 1
    #         print(f"Processing batch {count} at {time.time() - time_start} seconds")
            
    #         batch = batch.to(device)
    #         # Get node features and edge indices
    #         x_dict = batch.x_dict
    #         edge_index_dict = batch.edge_index_dict
            
    #         # Run forward pass through the model
    #         _, attention_weights = hetero_model(x_dict, edge_index_dict)
            
    #         # Get gene names from the batch data and transcripts
    #         # Convert tensor IDs to numpy array and then to list for indexing
    #         transcript_ids = batch['tx'].id.cpu().numpy()
            
    #         # Get gene names in the same order as the batch
    #         gene_names = [id_to_gene[id] for id in transcript_ids]
            
    #         # Extract attention weights into a structured dataset
    #         attention_df = extract_attention_df(attention_weights, gene_names)
            
    #         for layer_idx in range(layers):
    #             for head_idx in range(heads):
    #                 adj_matrix, count_matrix = summarize_attention_by_gene_df(attention_df, layer_idx=layer_idx, head_idx=head_idx, edge_type='tx-tx', gene_to_idx=gene_to_idx, visualize=False)
    #                 # update the attention_gene_matrix_dict
    #                 attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] += adj_matrix
    #                 attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx] += count_matrix
            
    # # compute the average attention weights
    
    # # Create figures directory if it doesn't exist
    # figures_dir = Path('figures')
    # if not figures_dir.exists():
    #     figures_dir.mkdir(parents=True, exist_ok=True)
    
    # for layer_idx in range(layers):
    #     for head_idx in range(heads):
    #         attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] = safe_divide_sparse_numpy(attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx], attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx])
            
    #         # visualize the attention weights
    #         plt.figure(figsize=(10, 8))
    #         sns.heatmap(attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx].toarray(), cmap='viridis', annot=False, cbar=True)
    #         plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}')
            
    #         plt.xlabel('Target Gene')
    #         plt.ylabel('Source Gene')
            
    #         # add gene labels
    #         # plt.xticks(np.arange(len(gene_to_idx)) + 0.5, sorted(gene_to_idx.keys()), rotation=90, ha='right')
    #         # plt.yticks(np.arange(len(gene_to_idx)) + 0.5, sorted(gene_to_idx.keys()), rotation=0)
            
    #         plt.tight_layout()
    #         plt.savefig(Path('figures') / f'Gene_attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png')
    #         plt.close()

    # # Generate comparison plots between layers and heads
    # # Compare all layers for each head
    # compare_attention_patterns(
    #     attention_gene_matrix_dict, 
    #     comparison_type='layers', 
    #     edge_type=edge_type,
    #     top_k=15,
    #     gene_to_idx=gene_to_idx
    # )
    
    # # Compare all heads for each layer
    # compare_attention_patterns(
    #     attention_gene_matrix_dict, 
    #     comparison_type='heads', 
    #     edge_type=edge_type,
    #     top_k=15,
    #     gene_to_idx=gene_to_idx
    # )

    # # Create results directory if it doesn't exist
    # results_dir = Path('results')
    # if not results_dir.exists():
    #     results_dir.mkdir(parents=True, exist_ok=True)
    
    # # Then save your file
    # with open(results_dir / 'attention_gene_matrix_dict.pkl', 'wb') as f:
    #     pickle.dump(attention_gene_matrix_dict, f)

    

if __name__ == '__main__':
    main()