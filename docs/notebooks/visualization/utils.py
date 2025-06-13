import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from sklearn.metrics import jaccard_score

@dataclass
class VisualizationConfig:
    """Configuration for attention visualization."""
    edge_type: str = 'tx-tx'
    cmap: str = 'viridis'
    cbar_label: str = 'Attention Weight'
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 300
    gene_types_dict: Optional[Dict[str, str]] = None
    cell_to_idx: Optional[Dict[str, int]] = None
    cell_order: Optional[List[str]] = None
    cell_type_to_color: Optional[Dict[str, str]] = None
    gene_type_to_color: Optional[Dict[str, str]] = None

@dataclass
class ComparisonConfig:
    """Configuration for attention pattern comparison."""
    comparison_type: str = 'layers'  # 'layers' or 'heads'
    layer_indices: Optional[List[int]] = None
    head_indices: Optional[List[int]] = None
    edge_type: str = 'tx-tx'
    selected_gene_names: Optional[List[str]] = None
    selected_gene_indices: Optional[List[int]] = None
    gene_to_idx: Optional[Dict[str, int]] = None
    cell_to_idx: Optional[Dict[str, int]] = None
    gene_types_dict: Optional[Dict[str, str]] = None
    figures_dir: Optional[Path] = None

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

def create_figures_dir(edge_type: str) -> Path:
    """Create figures directory if it doesn't exist."""
    figures_dir = Path('figures') / edge_type
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / 'dataset_visualization').mkdir(exist_ok=True)
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

class AttentionMatrixProcessor:
    """Handles processing of attention matrices."""
    
    @staticmethod
    def compute_attention_metrics(matrix: np.ndarray, threshold: float = 1e-5) -> Dict[str, float]:
        """Compute various metrics for an attention matrix."""
        return {
            'mean': np.mean(matrix),
            'std': np.std(matrix),
            'sparsity': np.mean(np.abs(matrix) < threshold),
            'entropy': entropy(matrix.flatten() + 1e-10)
        }
    
    @staticmethod
    def compute_difference_matrices(matrices: List[np.ndarray], mode: str = 'subsequent') -> List[np.ndarray]:
        """Compute difference matrices between attention matrices."""
        if mode == 'subsequent':
            return [matrices[i+1] - matrices[i] for i in range(len(matrices)-1)]
        elif mode == 'first':
            return [matrices[i] - matrices[0] for i in range(1, len(matrices))]
        raise ValueError("Mode must be 'subsequent' or 'first'")
    
    @staticmethod
    def compute_jaccard_similarity(matrices: List[np.ndarray], threshold: float = 1e-5) -> np.ndarray:
        """Compute Jaccard similarity between attention matrices."""
        n = len(matrices)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                m1 = (np.abs(matrices[i]) > threshold).astype(int)
                m2 = (np.abs(matrices[j]) > threshold).astype(int)
                similarity[i, j] = jaccard_score(m1.flatten(), m2.flatten())
        
        return similarity

class GeneTypeVisualizer:
    """Handles gene type visualization aspects."""
    
    @staticmethod
    def create_gene_type_colors(gene_types_dict: Dict[str, str]) -> Dict[str, Tuple[float, float, float]]:
        """Create color mapping for gene types."""
        unique_types = sorted(set(gene_types_dict.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        return dict(zip(unique_types, colors))
    
    @staticmethod
    def add_gene_type_coloring(ax, gene_names: List[str], gene_types_dict: Dict[str, str], 
                             type_to_color: Dict[str, Tuple[float, float, float]], axis: str = 'y'):
        """Add gene type coloring to axis labels."""
        labels = ax.get_yticklabels() if axis == 'y' else ax.get_xticklabels()
        gene_types = [gene_types_dict.get(gene, '') for gene in gene_names]
        
        for label, gene_type in zip(labels, gene_types):
            color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))
            highlight_color = (*color[:3], 0.3)
            label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
    
    @staticmethod
    def add_gene_type_legend(ax, type_to_color: Dict[str, Tuple[float, float, float]]):
        """Add gene type legend to plot."""
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=gene_type)
                         for gene_type, color in type_to_color.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

class AttentionMatrixAggregator:
    """Handles aggregation of attention matrices across layers and heads."""
    
    @staticmethod
    def compute_layer_average_attention(attention_gene_matrix_dict: Dict, layer_idx: int) -> np.ndarray:
        """Compute average attention weights across all heads for a given layer."""
        n_heads = len(attention_gene_matrix_dict['adj_matrix'][layer_idx])
        avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][layer_idx][0].toarray())
        
        for head_idx in range(n_heads):
            avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / n_heads
    
    @staticmethod
    def compute_head_average_attention(attention_gene_matrix_dict: Dict, head_idx: int) -> np.ndarray:
        """Compute average attention weights across all layers for a given head."""
        n_layers = len(attention_gene_matrix_dict['adj_matrix'])
        avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][head_idx].toarray())
        
        for layer_idx in range(n_layers):
            avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / n_layers
    
    @staticmethod
    def compute_overall_average_attention(attention_gene_matrix_dict: Dict) -> np.ndarray:
        """Compute average attention weights across all layers and heads."""
        n_layers = len(attention_gene_matrix_dict['adj_matrix'])
        n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
        avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray())
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / (n_layers * n_heads)

def get_top_genes_by_attention(attention_matrices, gene_names, top_k=20, mode='mean_row_col'):
    """
    Get top K genes by average attention (row+col mean or other mode).
    Parameters:
        attention_matrices: list of 2D numpy arrays or 3D numpy array (layers x genes x genes)
        gene_names: list of gene names
        top_k: number of top genes to return
        mode: 'mean_row_col' (default), or 'mean_row', 'mean_col'
    Returns:
        top_gene_names, top_gene_indices
    """
    if isinstance(attention_matrices, list):
        mats = np.stack([m if not hasattr(m, 'toarray') else m.toarray() for m in attention_matrices], axis=0)
    else:
        mats = attention_matrices
        
    if mode == 'mean_row_col':
        avg_attention = np.mean(mats, axis=0)
        avg = np.mean(avg_attention, axis=1) + np.mean(avg_attention, axis=0)
    elif mode == 'mean_row':
        avg_attention = np.mean(mats, axis=0)
        avg = np.mean(avg_attention, axis=1)
    elif mode == 'mean_col':
        avg_attention = np.mean(mats, axis=0)
        avg = np.mean(avg_attention, axis=0)
    else:
        raise ValueError('Unknown mode for get_top_genes_by_attention')
        
    top_indices = np.argsort(avg)[-top_k:]
    top_gene_names = [gene_names[i] for i in top_indices]
    return top_gene_names, top_indices
    
def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
