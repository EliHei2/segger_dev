import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import pandas as pd
from pathlib import Path
from .utils import create_figures_dir
"""
This function performs dimension reduction on gene attention patterns using various methods.

Parameters:
    attention_matrix : scipy.sparse.lil_matrix or numpy.ndarray
        The attention matrix of shape (num_genes, num_genes)
    method : str
        Dimension reduction method to use ('umap', 'pca', 'tsne')
    visualization : bool
        Whether to create and return a visualization plot
    random_state : int
        Random seed for reproducibility
        
Returns:
    - gene_coordinates : numpy.ndarray of shape (num_genes, 2)
    
"""
def gene_embedding(attention_matrix, method='umap', visualization=True, random_state=42):
    """
    Perform dimension reduction on gene attention patterns using various methods.
    
    Parameters:
    -----------
    attention_matrix : scipy.sparse.lil_matrix or numpy.ndarray
        The attention matrix of shape (num_genes, num_genes)
    method : str
        Dimension reduction method to use ('umap', 'pca', 'tsne')
    visualization : bool
        Whether to create and return a visualization plot
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        - gene_coordinates : numpy.ndarray of shape (num_genes, 2)
        - fig : matplotlib.figure.Figure (if visualization=True)
    """
    # Convert sparse matrix to dense if necessary
    if isinstance(attention_matrix, (lil_matrix)):
        attention_matrix = attention_matrix.toarray()
    
    # Create AnnData object
    adata = sc.AnnData(attention_matrix)
    
    # Always perform PCA preprocessing first
    sc.pp.pca(adata, n_comps=50, random_state=random_state)
    
    # Perform dimension reduction
    if method.lower() == 'umap':
        sc.pp.neighbors(adata, n_neighbors=15, random_state=random_state, use_rep='X_pca')
        sc.tl.umap(adata, random_state=random_state)
        coordinates = adata.obsm['X_umap']
    elif method.lower() == 'pca':
        coordinates = adata.obsm['X_pca'][:, :2]  # Take first 2 components
    elif method.lower() == 'tsne':
        sc.tl.tsne(adata, n_pcs=30, random_state=random_state, use_rep='X_pca')
        coordinates = adata.obsm['X_tsne']
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'umap', 'pca', or 'tsne'")
    
    if visualization:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.6)
        ax.set_title(f'Gene Embedding using {method.upper()}')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        plt.tight_layout()
        plt.savefig(f'{method}_gene_embedding.png')
        plt.close()
        return coordinates
    
    return coordinates

def visualize_attention_embedding(attention_gene_matrix_dict, method='umap', layer_idx=0, head_idx=0, 
                                top_k_genes=20, random_state=42):
    """
    Create an enhanced visualization of gene embeddings with attention patterns.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing attention matrices
    method : str
        Dimension reduction method to use
    layer_idx : int
        Index of the attention layer to use
    head_idx : int
        Index of the attention head to use
    top_k_genes : int
        Number of top genes to highlight in the plot
    random_state : int
        Random seed for reproducibility
    """
    # Get the attention matrix for the specified layer and head
    attention_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
    
    # Convert to dense array if it's sparse to avoid efficiency warnings
    if isinstance(attention_matrix, (lil_matrix)):
        attention_matrix = attention_matrix.toarray()
    
    # Perform dimension reduction
    coordinates = gene_embedding(attention_matrix, method=method, visualization=False, random_state=random_state)
    
    # Calculate gene importance (sum of attention weights)
    gene_importance = np.array(attention_matrix.sum(axis=1)).flatten()
    top_genes_idx = np.argsort(gene_importance)[-top_k_genes:]
    
    # Create figures directory
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'{method}_gene_embedding_layer_{layer_idx+1}_head_{head_idx+1}.png'
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all genes
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                        alpha=0.3, c=gene_importance, cmap='viridis')
    
    # Highlight top genes
    ax.scatter(coordinates[top_genes_idx, 0], coordinates[top_genes_idx, 1],
              c='red', alpha=0.6, s=100, label=f'Top {top_k_genes} genes')
    
    # Add colorbar
    plt.colorbar(scatter, label='Attention Weight Sum')
    
    # Customize plot
    ax.set_title(f'Gene Embedding using {method.upper()}\nLayer {layer_idx+1}, Head {head_idx+1}')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return fig 

def visualize_all_embeddings(attention_gene_matrix_dict, method='umap', top_k_genes=20, random_state=42):
    """
    Create a grid visualization of gene embeddings for all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing attention matrices
    method : str
        Dimension reduction method to use
    top_k_genes : int
        Number of top genes to highlight in each plot
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all embedding plots
    """
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Create figures directory
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'all_{method}_gene_embeddings.png'
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5*n_heads, 5*n_layers))
    axes = axes.flatten()
    
    # Generate embeddings for each layer and head
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Get the attention matrix
            attention_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
            
            # Convert to dense array if sparse
            if isinstance(attention_matrix, (lil_matrix)):
                attention_matrix = attention_matrix.toarray()
            
            # Perform dimension reduction
            coordinates = gene_embedding(attention_matrix, method=method, visualization=False, random_state=random_state)
            
            # Calculate gene importance
            gene_importance = np.array(attention_matrix.sum(axis=1)).flatten()
            top_genes_idx = np.argsort(gene_importance)[-top_k_genes:]
            
            # Get current subplot
            ax = axes[layer_idx * n_heads + head_idx]
            
            # Plot all genes
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                               alpha=0.3, c=gene_importance, cmap='viridis')
            
            # Highlight top genes
            ax.scatter(coordinates[top_genes_idx, 0], coordinates[top_genes_idx, 1],
                      c='red', alpha=0.6, s=50, label=f'Top {top_k_genes}')
            
            # Customize subplot
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.legend(fontsize='small')
    
    # Add colorbar to the last subplot
    plt.colorbar(scatter, ax=axes[-1], label='Attention Weight Sum')
    
    # Adjust layout
    plt.suptitle(f'Gene Embeddings using {method.upper()}\nAcross All Layers and Heads', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig 

def visualize_average_embedding(attention_gene_matrix_dict, gene_names, method='umap', random_state=42):
    """
    Create a visualization of gene embeddings using the average attention map across all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing attention matrices
    gene_names : list or array-like
        List of gene names corresponding to the attention matrix
    method : str
        Dimension reduction method to use (default: 'umap')
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the embedding plot with gene labels
    """
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Initialize average attention matrix
    avg_attention = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray())
    
    # Average attention matrices across all layers and heads
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attention_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
            if isinstance(attention_matrix, (lil_matrix)):
                attention_matrix = attention_matrix.toarray()
            avg_attention += attention_matrix
    
    avg_attention /= (n_layers * n_heads)
    
    # Perform dimension reduction
    coordinates = gene_embedding(avg_attention, method=method, visualization=False, random_state=random_state)
    
    # Calculate gene importance (sum of attention weights)
    gene_importance = np.array(avg_attention.sum(axis=1)).flatten()
    
    # Get indices of top and bottom 10 genes
    top_10_idx = np.argsort(gene_importance)[-10:]
    bottom_10_idx = np.argsort(gene_importance)[:10]
    
    # Create figures directory
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'average_{method}_gene_embedding.png'
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all genes
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                        alpha=0.3, c=gene_importance, cmap='viridis')
    
    # Highlight and label top 10 genes
    ax.scatter(coordinates[top_10_idx, 0], coordinates[top_10_idx, 1],
              c='red', alpha=0.6, s=100, label='Top 10 genes')
    for idx in top_10_idx:
        ax.annotate(gene_names[idx], 
                   (coordinates[idx, 0], coordinates[idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='red')
    
    # Highlight and label bottom 10 genes
    ax.scatter(coordinates[bottom_10_idx, 0], coordinates[bottom_10_idx, 1],
              c='blue', alpha=0.6, s=100, label='Bottom 10 genes')
    for idx in bottom_10_idx:
        ax.annotate(gene_names[idx], 
                   (coordinates[idx, 0], coordinates[idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='blue')
    
    # Add colorbar
    plt.colorbar(scatter, label='Average Attention Weight Sum')
    
    # Customize plot
    ax.set_title(f'Average Gene Embedding using {method.upper()}\nAcross All Layers and Heads')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return fig 