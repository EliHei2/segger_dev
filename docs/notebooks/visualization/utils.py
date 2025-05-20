import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, List, Tuple

def safe_divide_sparse_numpy(A, B):
    """Safely divide two sparse matrices, handling zero division."""
    A_coo = A.tocoo()
    B_coo = B.tocoo()
    
    result = A_coo.copy()
    mask = B_coo.data > 0
    result.data[mask] = A_coo.data[mask] / B_coo.data[mask]
    result.data[~mask] = 0
    
    result = result.tolil()
    return result

def downsample_matrix(matrix, target_size):
    """Downsample a matrix to the target size using averaging."""
    from skimage.measure import block_reduce
    
    h, w = matrix.shape
    if max(h, w) <= target_size:
        return matrix
    
    factor = max(1, int(max(h, w) / target_size))
    new_h = max(1, h // factor)
    new_w = max(1, w // factor)
    
    return block_reduce(matrix, block_size=(factor, factor), func=np.mean)

def create_figures_dir():
    """Create figures directory if it doesn't exist."""
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir

def plot_heatmap(matrix, title, xlabel, ylabel, save_path, cmap='viridis', vmin=None, vmax=None):
    """Generic function to plot a heatmap."""
    plt.figure(figsize=(12, 10))
    
    if max(matrix.shape) > 200:
        plt.ioff()
        plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        sns.heatmap(matrix, cmap=cmap, annot=False, cbar=True, vmin=vmin, vmax=vmax)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all') 
    
def load_attention_data(results_dir: Path = Path('results')) -> Dict:
    """Load the attention gene matrix dictionary from pickle file."""
    with open(results_dir / 'attention_gene_matrix_dict969.pkl', 'rb') as f:
        return pickle.load(f)
    
def get_top_genes_across_all_layers(
    attention_gene_matrix_dict: Dict,
    gene_names: List[str],
    top_k: int
) -> Tuple[List[str], np.ndarray]:
    """Get top K genes with highest average attention weights across all layers and heads."""
    n_layers = len(attention_gene_matrix_dict["adj_matrix"])
    n_heads = len(attention_gene_matrix_dict["adj_matrix"][0])
    
    # Calculate average attention weight for each gene across all layers and heads
    avg_attention = np.zeros(len(gene_names))
    count = 0
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attention_matrix = attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx]
            avg_attention += np.mean(attention_matrix.toarray(), axis=1)
            count += 1
    
    avg_attention /= count
    
    # Get indices of top K genes
    top_indices = np.argsort(avg_attention)[-top_k:]
    # Get corresponding gene names
    top_genes = [gene_names[i] for i in top_indices]
    
    return top_genes, top_indices
    
