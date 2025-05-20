import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_heatmap, create_figures_dir
from scipy.stats import entropy
from sklearn.metrics import jaccard_score
"""
The first function `summarize_attention_by_gene_df` summarizes attention weights and creates a heatmap of the attention weights at the gene level.

Parameters:
    attention_df : pandas.DataFrame
        DataFrame containing attention weights with columns:
        - 'layer': Layer index (1-indexed)
        - 'head': Head index (1-indexed)
        - 'source_gene': Source gene name
        - 'target_gene': Target gene name
        - 'attention_weight': Attention weight
    layer_idx : int
        Layer index (1-indexed)
    head_idx : int
        Head index (1-indexed)
    edge_type : str
        Edge type (e.g., 'tx-tx')
    gene_to_idx : dict
        Dictionary mapping gene names to their indices
    visualize : bool
        Whether to create and save a visualization plot
        
Returns:
    - adj_matrix : scipy.sparse.lil_matrix
        Adjacency matrix of shape (num_genes, num_genes)
    - count_matrix : scipy.sparse.lil_matrix
        Count matrix of shape (num_genes, num_genes)

The second function `compare_attention_patterns` creates comparison plots between different attention layers or heads.

Parameters:
    attention_gene_matrix_dict : dict
        Dictionary containing adjacency and count matrices for each layer and head
    comparison_type : str
        Type of comparison to perform ('layers' or 'heads')
    layer_indices : list
        List of layer indices to compare
    head_indices : list
        List of head indices to compare
    edge_type : str
        Edge type (e.g., 'tx-tx')
    top_k : int
        Number of top attention pairs to display in the comparison plots
    gene_to_idx : dict
        Dictionary mapping gene names to their indices
        
Returns:
    - None

"""
def summarize_attention_by_gene_df(attention_df, layer_idx, head_idx, edge_type, gene_to_idx, visualize=True):
    """Summarize attention weights at the gene level."""
    filtered_df = attention_df[
        (attention_df['layer'] == layer_idx + 1) & 
        (attention_df['head'] == head_idx + 1)
    ]
    
    num_genes = len(gene_to_idx)
    adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
    count_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
    
    for _, row in filtered_df.iterrows():
        src_idx = gene_to_idx[row['source_gene']]
        dst_idx = gene_to_idx[row['target_gene']]
        adj_matrix[src_idx, dst_idx] += row['attention_weight']
        count_matrix[src_idx, dst_idx] += 1
    
    adj_matrix = adj_matrix.tolil()
    count_matrix = count_matrix.tolil()
    
    if visualize:
        figures_dir = create_figures_dir()
        save_path = figures_dir / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png'
        
        plot_heatmap(
            matrix=adj_matrix.toarray()/count_matrix.toarray(),
            title=f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}',
            xlabel='Target Gene',
            ylabel='Source Gene',
            save_path=save_path
        )
    
    return adj_matrix, count_matrix

def compare_attention_patterns(attention_gene_matrix_dict, comparison_type='layers', 
                             layer_indices=None, head_indices=None, edge_type='tx-tx', 
                             top_k=20, gene_to_idx=None):
    """
    Create comparison plots between different attention layers or heads.
    1. Original heatmap comparison
    2. Difference matrices
    3. Attention metrics
    4. Jaccard similarity
    
    Parameters:
        attention_gene_matrix_dict : dict
            Dictionary containing adjacency and count matrices for each layer and head
        comparison_type : str
            Type of comparison to perform ('layers' or 'heads')
        layer_indices : list
    
    """
    figures_dir = create_figures_dir()
    layers = len(attention_gene_matrix_dict['adj_matrix'])
    heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    if layer_indices is None:
        layer_indices = list(range(layers))
    if head_indices is None:
        head_indices = list(range(heads))
    
    if gene_to_idx:
        idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    
    if comparison_type == 'layers':
        _compare_layers(attention_gene_matrix_dict, layer_indices, head_indices, 
                       edge_type, top_k, gene_to_idx, idx_to_gene, figures_dir)
    elif comparison_type == 'heads':
        _compare_heads(attention_gene_matrix_dict, layer_indices, head_indices, 
                      edge_type, top_k, gene_to_idx, idx_to_gene, figures_dir)
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}. Must be 'layers' or 'heads'.")

def _compare_layers(attention_gene_matrix_dict, layer_indices, head_indices, 
                   edge_type, top_k, gene_to_idx, idx_to_gene, figures_dir):
    """Helper function to compare attention patterns across layers."""
    for head_idx in head_indices:
        # Get matrices for this head across layers
        matrices = [attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray() 
                   for layer_idx in layer_indices]
        
        # 1. Original heatmap comparison
        fig, axes = plt.subplots(1, len(layer_indices), figsize=(6*len(layer_indices), 5))
        if len(layer_indices) == 1:
            axes = [axes]
            
        for i, matrix in enumerate(matrices):
            im = sns.heatmap(matrix, cmap='viridis', ax=axes[i], cbar=(i == len(layer_indices)-1))
            axes[i].set_title(f'Layer {layer_indices[i]+1}, Head {head_idx+1}')
            axes[i].set_xlabel('Target Gene')
            axes[i].set_ylabel('Source Gene')
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'layer_comparison_head_{head_idx+1}_{edge_type}.png')
        plt.close()
        
        # 2. Difference matrices
        for mode in ['subsequent', 'first']:
            diff_matrices = compute_difference_matrices(matrices, mode=mode)
            fig, axes = plt.subplots(1, len(diff_matrices), figsize=(6*len(diff_matrices), 5))
            if len(diff_matrices) == 1:
                axes = [axes]
                
            for i, diff_matrix in enumerate(diff_matrices):
                im = sns.heatmap(diff_matrix, cmap='RdBu_r', center=0, ax=axes[i], 
                               cbar=(i == len(diff_matrices)-1))
                if mode == 'subsequent':
                    title = f'Layer {layer_indices[i+1]+1} - Layer {layer_indices[i]+1}'
                else:
                    title = f'Layer {layer_indices[i+1]+1} - Layer {layer_indices[0]+1}'
                axes[i].set_title(f'{title}, Head {head_idx+1}')
                axes[i].set_xlabel('Target Gene')
                axes[i].set_ylabel('Source Gene')
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'layer_differences_{mode}_head_{head_idx+1}_{edge_type}.png')
            plt.close()
        
        # 3. Attention metrics
        metrics = [compute_attention_metrics(matrix) for matrix in matrices]
        metrics_df = pd.DataFrame(metrics, index=[f'Layer {i+1}' for i in layer_indices])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_df['mean'].plot(ax=axes[0,0], marker='o')
        axes[0,0].set_title('Mean Attention Weight')
        axes[0,0].set_xlabel('Layer')
        axes[0,0].set_ylabel('Mean')
        
        metrics_df['std'].plot(ax=axes[0,1], marker='o')
        axes[0,1].set_title('Standard Deviation')
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Std')
        
        metrics_df['sparsity'].plot(ax=axes[1,0], marker='o')
        axes[1,0].set_title('Sparsity')
        axes[1,0].set_xlabel('Layer')
        axes[1,0].set_ylabel('Sparsity')
        
        metrics_df['entropy'].plot(ax=axes[1,1], marker='o')
        axes[1,1].set_title('Entropy')
        axes[1,1].set_xlabel('Layer')
        axes[1,1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'layer_metrics_head_{head_idx+1}_{edge_type}.png')
        plt.close()
        
        # 4. Jaccard similarity
        similarity = compute_jaccard_similarity(matrices)
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity, annot=True, cmap='viridis',
                   xticklabels=[f'Layer {i+1}' for i in layer_indices],
                   yticklabels=[f'Layer {i+1}' for i in layer_indices])
        plt.title(f'Jaccard Similarity Between Layers (Head {head_idx+1})')
        plt.tight_layout()
        plt.savefig(figures_dir / f'layer_jaccard_head_{head_idx+1}_{edge_type}.png')
        plt.close()

def _compare_heads(attention_gene_matrix_dict, layer_indices, head_indices, 
                  edge_type, top_k, gene_to_idx, idx_to_gene, figures_dir):
    """Helper function to compare attention patterns across heads."""
    for layer_idx in layer_indices:
        # Get matrices for this layer across heads
        matrices = [attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray() 
                   for head_idx in head_indices]
        
        # 1. Original heatmap comparison
        fig, axes = plt.subplots(1, len(head_indices), figsize=(6*len(head_indices), 5))
        if len(head_indices) == 1:
            axes = [axes]
            
        for i, matrix in enumerate(matrices):
            im = sns.heatmap(matrix, cmap='viridis', ax=axes[i], cbar=(i == len(head_indices)-1))
            axes[i].set_title(f'Layer {layer_idx+1}, Head {head_indices[i]+1}')
            axes[i].set_xlabel('Target Gene')
            axes[i].set_ylabel('Source Gene')
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'head_comparison_layer_{layer_idx+1}_{edge_type}.png')
        plt.close()
        
        # 2. Difference matrices
        for mode in ['subsequent', 'first']:
            diff_matrices = compute_difference_matrices(matrices, mode=mode)
            fig, axes = plt.subplots(1, len(diff_matrices), figsize=(6*len(diff_matrices), 5))
            if len(diff_matrices) == 1:
                axes = [axes]
                
            for i, diff_matrix in enumerate(diff_matrices):
                im = sns.heatmap(diff_matrix, cmap='RdBu_r', center=0, ax=axes[i], 
                               cbar=(i == len(diff_matrices)-1))
                if mode == 'subsequent':
                    title = f'Head {head_indices[i+1]+1} - Head {head_indices[i]+1}'
                else:
                    title = f'Head {head_indices[i+1]+1} - Head {head_indices[0]+1}'
                axes[i].set_title(f'{title}, Layer {layer_idx+1}')
                axes[i].set_xlabel('Target Gene')
                axes[i].set_ylabel('Source Gene')
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'head_differences_{mode}_layer_{layer_idx+1}_{edge_type}.png')
            plt.close()
        
        # 3. Attention metrics
        metrics = [compute_attention_metrics(matrix) for matrix in matrices]
        metrics_df = pd.DataFrame(metrics, index=[f'Head {i+1}' for i in head_indices])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_df['mean'].plot(ax=axes[0,0], marker='o')
        axes[0,0].set_title('Mean Attention Weight')
        axes[0,0].set_xlabel('Head')
        axes[0,0].set_ylabel('Mean')
        
        metrics_df['std'].plot(ax=axes[0,1], marker='o')
        axes[0,1].set_title('Standard Deviation')
        axes[0,1].set_xlabel('Head')
        axes[0,1].set_ylabel('Std')
        
        metrics_df['sparsity'].plot(ax=axes[1,0], marker='o')
        axes[1,0].set_title('Sparsity')
        axes[1,0].set_xlabel('Head')
        axes[1,0].set_ylabel('Sparsity')
        
        metrics_df['entropy'].plot(ax=axes[1,1], marker='o')
        axes[1,1].set_title('Entropy')
        axes[1,1].set_xlabel('Head')
        axes[1,1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'head_metrics_layer_{layer_idx+1}_{edge_type}.png')
        plt.close()
        
        # 4. Jaccard similarity
        similarity = compute_jaccard_similarity(matrices)
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity, annot=True, cmap='viridis',
                   xticklabels=[f'Head {i+1}' for i in head_indices],
                   yticklabels=[f'Head {i+1}' for i in head_indices])
        plt.title(f'Jaccard Similarity Between Heads (Layer {layer_idx+1})')
        plt.tight_layout()
        plt.savefig(figures_dir / f'head_jaccard_layer_{layer_idx+1}_{edge_type}.png')
        plt.close()

def compute_attention_metrics(matrix, threshold=1e-5):
    """Compute various metrics for an attention matrix.
    
    Args:
        matrix: numpy array of attention weights
        threshold: threshold for sparsity calculation
        
    Returns:
        dict containing mean, std, sparsity, and entropy
    """
    metrics = {
        'mean': np.mean(matrix),
        'std': np.std(matrix),
        'sparsity': np.mean(np.abs(matrix) < threshold),
        'entropy': entropy(matrix.flatten() + 1e-10)  # Add small constant to avoid log(0)
    }
    return metrics

def compute_difference_matrices(matrices, mode='subsequent'):
    """Compute difference matrices between attention matrices.
    
    Args:
        matrices: list of attention matrices
        mode: 'subsequent' for differences between consecutive matrices,
              'first' for differences from the first matrix
    
    Returns:
        list of difference matrices
    """
    if mode == 'subsequent':
        return [matrices[i+1] - matrices[i] for i in range(len(matrices)-1)]
    elif mode == 'first':
        return [matrices[i] - matrices[0] for i in range(1, len(matrices))]
    else:
        raise ValueError("Mode must be 'subsequent' or 'first'")

def compute_jaccard_similarity(matrices, threshold=1e-5):
    """Compute Jaccard similarity between attention matrices.
    
    Args:
        matrices: list of attention matrices
        threshold: threshold for binarizing attention weights
        
    Returns:
        similarity matrix of shape (n_matrices, n_matrices)
    """
    n = len(matrices)
    similarity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Binarize matrices
            m1 = (np.abs(matrices[i]) > threshold).astype(int)
            m2 = (np.abs(matrices[j]) > threshold).astype(int)
            
            # Compute Jaccard similarity
            similarity[i, j] = jaccard_score(m1.flatten(), m2.flatten())
    
    return similarity

def visualize_all_attention_patterns(attention_gene_matrix_dict, edge_type='tx-tx', gene_to_idx=None):
    """
    Create a grid visualization of attention patterns for all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing adjacency and count matrices for each layer and head
    edge_type : str
        Edge type (e.g., 'tx-tx')
    gene_to_idx : dict
        Dictionary mapping gene names to their indices
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all attention pattern heatmaps
    """
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Create figures directory
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'all_attention_patterns_{edge_type}.png'
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5*n_heads, 5*n_layers))
    axes = axes.flatten()
    
    # Generate heatmaps for each layer and head
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Get the attention matrix
            adj_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
            count_matrix = attention_gene_matrix_dict['count_matrix'][layer_idx][head_idx]
            
            # Convert to dense array if sparse
            if isinstance(adj_matrix, (lil_matrix)):
                adj_matrix = adj_matrix.toarray()
            if isinstance(count_matrix, (lil_matrix)):
                count_matrix = count_matrix.toarray()
            
            # Compute normalized attention weights
            normalized_matrix = adj_matrix / (count_matrix + 1e-10)  # Add small constant to avoid division by zero
            
            # Get current subplot
            ax = axes[layer_idx * n_heads + head_idx]
            
            # Create heatmap
            im = sns.heatmap(normalized_matrix, cmap='viridis', ax=ax, cbar=False)
            
            # Customize subplot
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
            ax.set_xlabel('Target Gene')
            ax.set_ylabel('Source Gene')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
    
    # Add colorbar to the last subplot
    plt.colorbar(im.collections[0], ax=axes[-1], label='Normalized Attention Weight')
    
    # Adjust layout
    plt.suptitle(f'Attention Patterns Across All Layers and Heads\nEdge Type: {edge_type}', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig

def visualize_all_attention_patterns_with_metrics(attention_gene_matrix_dict, edge_type='tx-tx', gene_to_idx=None):
    """
    Create a comprehensive visualization of attention patterns and their metrics for all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing adjacency and count matrices for each layer and head
    edge_type : str
        Edge type (e.g., 'tx-tx')
    gene_to_idx : dict
        Dictionary mapping gene names to their indices
        
    Returns:
    --------
    tuple
        - fig_patterns : matplotlib.figure.Figure for attention patterns
        - fig_metrics : matplotlib.figure.Figure for attention metrics
    """
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Create figures directory
    figures_dir = create_figures_dir()
    
    # 1. Create attention patterns grid
    fig_patterns = visualize_all_attention_patterns(attention_gene_matrix_dict, edge_type, gene_to_idx)
    
    # 2. Create metrics visualization
    metrics_data = {
        'mean': np.zeros((n_layers, n_heads)),
        'std': np.zeros((n_layers, n_heads)),
        'sparsity': np.zeros((n_layers, n_heads)),
        'entropy': np.zeros((n_layers, n_heads))
    }
    
    # Compute metrics for each layer and head
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            adj_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
            if isinstance(adj_matrix, (lil_matrix)):
                adj_matrix = adj_matrix.toarray()
            
            metrics = compute_attention_metrics(adj_matrix)
            for metric_name, value in metrics.items():
                metrics_data[metric_name][layer_idx, head_idx] = value
    
    # Create metrics visualization
    fig_metrics, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric_name, data) in enumerate(metrics_data.items()):
        im = sns.heatmap(data, cmap='viridis', ax=axes[i], cbar=True,
                        xticklabels=[f'Head {i+1}' for i in range(n_heads)],
                        yticklabels=[f'Layer {i+1}' for i in range(n_layers)])
        axes[i].set_title(f'{metric_name.capitalize()} Across Layers and Heads')
        axes[i].set_xlabel('Head')
        axes[i].set_ylabel('Layer')
    
    plt.suptitle(f'Attention Metrics Across All Layers and Heads\nEdge Type: {edge_type}', y=1.02)
    plt.tight_layout()
    
    # Save metrics figure
    save_path = figures_dir / f'all_attention_metrics_{edge_type}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_patterns, fig_metrics 