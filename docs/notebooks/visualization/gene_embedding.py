import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import pandas as pd
from pathlib import Path
from .utils import create_figures_dir, hex_to_rgb
import umap
from sklearn.manifold import TSNE
from typing import Tuple

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
                                top_k_genes=20, gene_types_dict=None, gene_type_to_color=None, random_state=42):
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
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    gene_type_to_color : dict, optional
        Dictionary mapping gene types to colors
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
    
    if gene_types_dict is not None:
        # Create color mapping for gene types
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            # Convert hex colors to RGB tuples
            type_to_color = {t: hex_to_rgb(gene_type_to_color.get(t, '#808080')) for t in unique_types}
        else:
            # Fallback to default color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
        
        # Get gene types for all genes
        idx_to_gene = {idx: gene for gene, idx in gene_types_dict.items()}
        gene_types = [gene_types_dict.get(idx_to_gene[i], '') for i in range(len(gene_types_dict))]
        
        # Plot genes colored by type
        for gene_type in unique_types:
            type_mask = [t == gene_type for t in gene_types]
            if np.any(type_mask):
                ax.scatter(coordinates[type_mask, 0], coordinates[type_mask, 1],
                          c=[type_to_color[gene_type]], alpha=0.3, label=gene_type)
        
        # Create legend for gene types
        ax.legend(title='Gene Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Plot all genes with importance-based coloring
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           alpha=0.3, c=gene_importance, cmap='viridis')
    
    # Highlight top genes
    ax.scatter(coordinates[top_genes_idx, 0], coordinates[top_genes_idx, 1],
              c='red', alpha=0.6, s=100, label=f'Top {top_k_genes} genes')
    
    # Customize plot
    ax.set_title(f'Gene Embedding using {method.upper()}\nLayer {layer_idx+1}, Head {head_idx+1}')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return fig

def visualize_all_embeddings(attention_gene_matrix_dict, gene_names, method='umap', top_k_genes=20, 
                           gene_types_dict=None, gene_type_to_color=None, random_state=42):
    """
    Create a grid visualization of gene embeddings for all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing attention matrices
    gene_names : list
        List of gene names
    method : str
        Dimension reduction method to use
    top_k_genes : int
        Number of top genes to highlight in each plot
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    gene_type_to_color : dict, optional
        Dictionary mapping gene types to colors
    random_state : int
        Random seed for reproducibility
    """
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    
    # Create figures directory
    figures_dir = create_figures_dir(edge_type='tx-tx')
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
            
            if gene_types_dict is not None:
                # Create color mapping for gene types
                unique_types = sorted(set(gene_types_dict.values()))
                if gene_type_to_color is not None:
                    # Convert hex colors to RGB tuples
                    type_to_color = {t: hex_to_rgb(gene_type_to_color.get(t, '#808080')) for t in unique_types}
                else:
                    # Fallback to default color palette
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
                    type_to_color = dict(zip(unique_types, colors))
                
                # Get gene types for all genes
                gene_types = [gene_types_dict.get(gene, '') for gene in gene_names]
                
                # Plot genes colored by type
                for gene_type in unique_types:
                    type_mask = [t == gene_type for t in gene_types]
                    if np.any(type_mask):
                        ax.scatter(coordinates[type_mask, 0], coordinates[type_mask, 1],
                                  c=[type_to_color[gene_type]], alpha=0.3, label=gene_type)
                
                # Add legend for gene types if it's the first subplot
                if layer_idx == 0 and head_idx == 0:
                    ax.legend(title='Gene Types', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            else:
                # Plot all genes with importance-based coloring
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
    
    # Add colorbar to the last subplot if not using gene types
    if gene_types_dict is None:
        plt.colorbar(scatter, ax=axes[-1], label='Attention Weight Sum')
    
    # Adjust layout
    plt.suptitle(f'Gene Embeddings using {method.upper()}\nAcross All Layers and Heads', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig

def visualize_average_embedding(attention_gene_matrix_dict, gene_names, method='umap', 
                              gene_types_dict=None, gene_type_to_color=None):
    """
    Visualize the average embedding of genes across all layers and heads.
    
    Args:
        attention_gene_matrix_dict (dict): Dictionary containing attention matrices
        gene_names (list): List of gene names
        method (str): Embedding method ('umap' or 'tsne')
        gene_types_dict (dict): Dictionary mapping gene names to their types
        gene_type_to_color (dict): Dictionary mapping gene types to colors
    """
    # Compute average attention matrix
    avg_matrix = np.zeros((len(gene_names), len(gene_names)))
    count = 0
    
    for layer_idx in range(len(attention_gene_matrix_dict["adj_matrix"])):
        for head_idx in range(len(attention_gene_matrix_dict["adj_matrix"][0])):
            matrix = attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx].toarray()
            if matrix.shape == (len(gene_names), len(gene_names)):
                avg_matrix += matrix
                count += 1
    
    if count > 0:
        avg_matrix /= count
    
    # Compute embedding
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42)
    
    coordinates = reducer.fit_transform(avg_matrix)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    if gene_types_dict is not None:
        # Create color mapping for gene types
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            # Convert hex colors to RGB tuples
            type_to_color = {t: hex_to_rgb(gene_type_to_color.get(t, '#808080')) for t in unique_types}
        else:
            # Fallback to default color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
        
        # Get gene types for all genes
        gene_types = [gene_types_dict.get(gene, '') for gene in gene_names]
        
        # Plot genes colored by type
        for gene_type in unique_types:
            type_mask = [t == gene_type for t in gene_types]
            if np.any(type_mask):
                ax.scatter(coordinates[type_mask, 0], coordinates[type_mask, 1],
                           c=[type_to_color[gene_type]], alpha=0.3, label=gene_type)
    else:
        # Plot all genes in the same color if no gene types provided
        ax.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.3)
    
    plt.title(f'Average Gene Embedding ({method.upper()})')
    if gene_types_dict is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save figure
    figures_dir = create_figures_dir('tx-tx')
    plt.savefig(figures_dir / f'average_gene_embedding_{method}.png', dpi=300, bbox_inches='tight')
    plt.close() 