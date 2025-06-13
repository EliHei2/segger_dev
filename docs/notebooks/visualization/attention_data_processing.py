import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from utils import load_attention_data, get_top_genes_by_attention

def save_intermediate_data(
    attention_gene_matrix_dict: Dict,
    gene_names: List[str],
    output_dir: Path = Path('intermediate_data')
) -> None:
    """Save intermediate attention data for faster loading.
    
    Args:
        attention_gene_matrix_dict: Dictionary containing attention matrices
        gene_names: List of gene names
        output_dir: Directory to save intermediate data
    """
    output_dir.mkdir(exist_ok=True)
    
    # Save attention matrices
    with open(output_dir / 'attention_matrices.pkl', 'wb') as f:
        pickle.dump(attention_gene_matrix_dict, f)
    
    # Save gene names
    with open(output_dir / 'gene_names.pkl', 'wb') as f:
        pickle.dump(gene_names, f)
    
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict["adj_matrix"])
    n_heads = len(attention_gene_matrix_dict["adj_matrix"][0])
    
    # Pre-compute and save layer-wise average attention matrices
    layer_avg_matrices = []
    for l in range(n_layers):
        layer_matrix = np.zeros((len(gene_names), len(gene_names)))
        for h in range(n_heads):
            attention_matrix = attention_gene_matrix_dict["adj_matrix"][l][h]
            layer_matrix += attention_matrix.toarray()
        layer_matrix /= n_heads
        layer_avg_matrices.append(layer_matrix)
    np.save(output_dir / 'layer_avg_matrices.npy', layer_avg_matrices)
    
    # Pre-compute and save head-wise average attention matrices
    head_avg_matrices = []
    for h in range(n_heads):
        head_matrix = np.zeros((len(gene_names), len(gene_names)))
        for l in range(n_layers):
            attention_matrix = attention_gene_matrix_dict["adj_matrix"][l][h]
            head_matrix += attention_matrix.toarray()
        head_matrix /= n_layers
        head_avg_matrices.append(head_matrix)
    np.save(output_dir / 'head_avg_matrices.npy', head_avg_matrices)
    
    # Pre-compute and save overall average attention matrix
    avg_matrix = np.zeros((len(gene_names), len(gene_names)))
    for l in range(n_layers):
        for h in range(n_heads):
            attention_matrix = attention_gene_matrix_dict["adj_matrix"][l][h]
            avg_matrix += attention_matrix.toarray()
    avg_matrix /= (n_layers * n_heads)
    np.save(output_dir / 'average_attention_matrix.npy', avg_matrix)
    
    # Pre-compute and save top genes for different k values
    all_matrices = [m.toarray() for l in attention_gene_matrix_dict['adj_matrix'] for m in l]
    for k in [50]:
        top_genes, top_indices = get_top_genes_by_attention(all_matrices, gene_names, top_k=k, mode = "mean_row") # get the top k genes by how much they attend to other genes
        with open(output_dir / f'top_genes_k{k}.pkl', 'wb') as f:
            pickle.dump((top_genes, top_indices), f)

def load_intermediate_data(
    output_dir: Path = Path('intermediate_data')
) -> Tuple[Dict, List[str], np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[List[str], np.ndarray]]]:
    """Load intermediate attention data.
    
    Args:
        output_dir: Directory containing intermediate data
        
    Returns:
        Tuple containing:
        - attention_gene_matrix_dict: Dictionary containing attention matrices
        - gene_names: List of gene names
        - average_attention_matrix: Pre-computed average attention matrix
        - layer_avg_matrices: List of layer-wise average attention matrices
        - head_avg_matrices: List of head-wise average attention matrices
        - top_genes_dict: Dictionary mapping k to (top_genes, top_indices) tuples
    """
    # Load attention matrices
    with open(output_dir / 'attention_matrices.pkl', 'rb') as f:
        attention_gene_matrix_dict = pickle.load(f)
    
    # Load gene names
    with open(output_dir / 'gene_names.pkl', 'rb') as f:
        gene_names = pickle.load(f)
    
    # Load average attention matrices
    average_attention_matrix = np.load(output_dir / 'average_attention_matrix.npy')
    layer_avg_matrices = np.load(output_dir / 'layer_avg_matrices.npy')
    head_avg_matrices = np.load(output_dir / 'head_avg_matrices.npy')
    
    # Load top genes for different k values
    top_genes_dict = {}
    for k in [50]:
        with open(output_dir / f'top_genes_k{k}.pkl', 'rb') as f:
            top_genes_dict[k] = pickle.load(f)
    
    return attention_gene_matrix_dict, gene_names, average_attention_matrix, layer_avg_matrices, head_avg_matrices, top_genes_dict 