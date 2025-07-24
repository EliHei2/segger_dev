import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_heatmap, create_figures_dir, get_top_genes_by_attention, AttentionMatrixProcessor, GeneTypeVisualizer, VisualizationConfig, ComparisonConfig
from scipy.stats import entropy
from sklearn.metrics import jaccard_score
from pathlib import Path
import torch
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
The first function `summarize_attention_by_gene_df` summarizes attention weights and creates a heatmap of the attention weights at the gene level.

Parameters:
    attention_df : pandas.DataFrame
        DataFrame containing attention weights with columns:
        - 'layer': Layer index (1-indexed)
        - 'head': Head index (1-indexed)
        - 'source_gene': Source gene name
        - 'target_gene'/'target_cell_id': Target gene name/cell ID
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
    selected_gene_names : list
        List of gene names to visualize
    selected_gene_indices : list
        List of gene indices corresponding to selected_gene_names
    gene_to_idx : dict
        Dictionary mapping gene names to their indices
    cell_to_idx : dict
        Dictionary mapping cell types to their indices (required for tx-bd edge type)
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    cell_types_dict : dict, optional
        Dictionary mapping cell ids to their types
        
Returns:
    - None

"""
def summarize_attention_by_gene_df(attention_df, layer_idx, head_idx, edge_type, gene_to_idx = None, cell_to_idx = None):
    """
    Summarize attention weights at the gene level.
    
    Parameters:
        attention_df : pandas.DataFrame
            DataFrame containing attention weights with columns:
            - 'layer': Layer index (1-indexed)
            - 'head': Head index (1-indexed)
            - 'source_gene': Source gene name
            - 'target_gene'/'target_cell_id': Target gene name/cell ID
            - 'attention_weight': Attention weight
        layer_idx : int
            Layer index (1-indexed)
        head_idx : int
            Head index (1-indexed)
        edge_type : str
            Edge type (e.g., 'tx-tx', 'tx-bd')
        gene_to_idx : dict
            Dictionary mapping gene names to their indices
        cell_to_idx : dict
            Dictionary mapping cell IDs to their indices
        visualize : bool
            Whether to create and save a visualization plot
            
    Returns:
        - adj_matrix : scipy.sparse.lil_matrix
    """
    filtered_df = attention_df[
        (attention_df['layer'] == layer_idx + 1) & 
        (attention_df['head'] == head_idx + 1)
    ]
    
    num_genes = len(gene_to_idx)
        
    if edge_type == 'tx-tx':
        adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
        count_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
    elif edge_type == 'tx-bd':
        num_cells = len(cell_to_idx)
        adj_matrix = lil_matrix((num_genes, num_cells), dtype=np.float32)
        count_matrix = lil_matrix((num_genes, num_cells), dtype=np.float32)
    
    for _, row in filtered_df.iterrows():
        src_idx = gene_to_idx[row['source_gene']]
        if edge_type == 'tx-tx':
            dst_idx = gene_to_idx[row['target_gene']]
        elif edge_type == 'tx-bd':
            dst_idx = cell_to_idx[row['target_cell_id']]
        adj_matrix[src_idx, dst_idx] += row['attention_weight']
        count_matrix[src_idx, dst_idx] += 1
    
    adj_matrix = adj_matrix.tolil()
    count_matrix = count_matrix.tolil()
    
    return adj_matrix, count_matrix

def compute_layer_average_attention(attention_gene_matrix_dict, layer_idx):
    """Compute average attention weights across all heads for a given layer."""
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][layer_idx])
    avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][layer_idx][0].toarray())
    
    for head_idx in range(n_heads):
        avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
    
    return avg_matrix / n_heads

def compute_head_average_attention(attention_gene_matrix_dict, head_idx):
    """Compute average attention weights across all layers for a given head."""
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][head_idx].toarray())
    
    for layer_idx in range(n_layers):
        avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
    
    return avg_matrix / n_layers

def compute_overall_average_attention(attention_gene_matrix_dict):
    """Compute average attention weights across all layers and heads."""
    n_layers = len(attention_gene_matrix_dict['adj_matrix'])
    n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
    avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray())
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
    
    return avg_matrix / (n_layers * n_heads)



def plot_attention_matrix_with_gene_order(matrix, gene_names, gene_indices, ordered_gene_names, ordered_gene_indices, 
                                        title, figures_dir, cmap='viridis', cbar_label='Attention Weight',
                                        edge_type='tx-tx', cell_to_idx=None, gene_types_dict=None):
    """
    Plot an attention matrix using a predefined gene order.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Attention matrix to plot
    gene_names : list
        List of all gene names
    gene_indices : list
        List of all gene indices
    ordered_gene_names : list
        List of gene names in the desired order
    ordered_gene_indices : list
        List of gene indices in the desired order
    title : str
        Title for the plot
    figures_dir : Path
        Directory to save figures
    cmap : str
        Colormap to use
    cbar_label : str
        Label for the colorbar
    edge_type : str
        Edge type (e.g., 'tx-tx', 'tx-bd')
    cell_to_idx : dict
        Dictionary mapping cell types to their indices (required for tx-bd edge type)
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    """
    if edge_type == 'tx-tx':
        # Select only the chosen genes
        selected_matrix = matrix[np.ix_(gene_indices, gene_indices)]
        
        # Create DataFrame with ordered genes
        df = pd.DataFrame(selected_matrix, index=gene_names, columns=gene_names)
        
        # Reorder the DataFrame according to the predefined order
        df = df.reindex(index=ordered_gene_names, columns=ordered_gene_names)
        
        # Create heatmap
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(df, cmap=cmap, cbar_kws={'label': cbar_label})
        
        if gene_types_dict is not None:
            # Create color mapping for gene types
            unique_types = sorted(set(gene_types_dict.values()))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
            
            # Get gene types for source and target
            source_types = [gene_types_dict.get(gene, '') for gene in ordered_gene_names]
            target_types = source_types
            
            # Color the y-axis labels (source genes)
            y_labels = ax.get_yticklabels()
            for label, gene_type in zip(y_labels, source_types):
                color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))  # gray for unknown types
                highlight_color = (*color[:3], 0.3)  # 30% opacity
                label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
            
            # Color the x-axis labels (target genes)
            x_labels = ax.get_xticklabels()
            for label, gene_type in zip(x_labels, target_types):
                color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))  # gray for unknown types
                highlight_color = (*color[:3], 0.3)  # 30% opacity
                label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
            
            # Create legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=gene_type)
                             for gene_type, color in type_to_color.items()]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title(title)
        plt.xlabel('Target Gene')
        plt.ylabel('Source Gene')
        plt.tight_layout()
    else:  # tx-bd
        if cell_to_idx is None:
            raise ValueError("cell_to_idx must be provided for tx-bd edge type")
            
        # Select only the chosen genes
        selected_matrix = matrix[np.ix_(gene_indices, range(len(cell_to_idx)))]
        
        # Create DataFrame with ordered genes and cell IDs
        cell_ids = list(cell_to_idx.keys())
        df = pd.DataFrame(selected_matrix, index=gene_names, columns=cell_ids)
        
        # Reorder the DataFrame according to the predefined order
        df = df.reindex(index=ordered_gene_names)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(df, cmap=cmap, cbar_kws={'label': cbar_label})
        
        if gene_types_dict is not None:
            # Create color mapping for gene types
            unique_types = sorted(set(gene_types_dict.values()))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
            
            # Get gene types for source genes
            source_types = [gene_types_dict.get(gene, '') for gene in ordered_gene_names]
            
            # Color the y-axis labels (source genes)
            y_labels = ax.get_yticklabels()
            for label, gene_type in zip(y_labels, source_types):
                color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))  # gray for unknown types
                highlight_color = (*color[:3], 0.3)  # 30% opacity
                label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
            
            # Create legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=gene_type)
                             for gene_type, color in type_to_color.items()]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title(title)
        plt.xlabel('Target Cell ID')
        plt.ylabel('Source Gene')
        plt.tight_layout()
    
    # Save the plot
    plt.savefig(figures_dir / 'dataset_visualization' / f'{title.lower().replace(" ", "_")}.png', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_all_attention_patterns(attention_gene_matrix_dict: Dict, 
                                   selected_gene_names: Optional[List[str]] = None,
                                   selected_gene_indices: Optional[List[int]] = None,
                                   config: Optional[VisualizationConfig] = None):
    """
    Create a comprehensive visualization of attention patterns.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : Dict
        Dictionary containing adjacency matrices for each layer and head
    selected_gene_names : Optional[List[str]]
        List of gene names to visualize
    selected_gene_indices : Optional[List[int]]
        List of gene indices corresponding to selected_gene_names
    config : Optional[VisualizationConfig]
        Configuration for visualization
    """
    if config is None:
        config = VisualizationConfig()
    
    figures_dir = create_figures_dir(config.edge_type)
    visualizer = AttentionVisualizer(config)
    
    if selected_gene_names and selected_gene_indices:
        # Plot all layers and heads in one figure
        visualizer.plot_all_layers_heads(attention_gene_matrix_dict, selected_gene_names, 
                                       selected_gene_indices, figures_dir, cell_order=config.cell_order)
        
        # Visualize overall average attention pattern
        overall_avg = AttentionMatrixAggregator.compute_overall_average_attention(
            attention_gene_matrix_dict, selected_gene_indices, config.edge_type
        )
        visualizer.plot_attention_matrix(
            overall_avg, selected_gene_names, selected_gene_names,
            f'Overall Average Attention Pattern', figures_dir)
        
        # Visualize layer averages
        for layer_idx in range(len(attention_gene_matrix_dict['adj_matrix'])):
            layer_avg = AttentionMatrixAggregator.compute_layer_average_attention(
                attention_gene_matrix_dict, layer_idx, selected_gene_indices, config.edge_type
            )
            visualizer.plot_attention_matrix(
                layer_avg, selected_gene_names, selected_gene_names,
                f'Layer {layer_idx+1} Average Attention Pattern', figures_dir
            )
        
        # Visualize head averages
        for head_idx in range(len(attention_gene_matrix_dict['adj_matrix'][0])):
            head_avg = AttentionMatrixAggregator.compute_head_average_attention(
                attention_gene_matrix_dict, head_idx, selected_gene_indices, config.edge_type
            )
            visualizer.plot_attention_matrix(
                head_avg, selected_gene_names, selected_gene_names,
                f'Head {head_idx+1} Average Attention Pattern', figures_dir
            )
        
        # Visualize individual layer-head patterns
        for layer_idx in range(len(attention_gene_matrix_dict['adj_matrix'])):
            for head_idx in range(len(attention_gene_matrix_dict['adj_matrix'][0])):
                adj_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
                if isinstance(adj_matrix, lil_matrix):
                    adj_matrix = adj_matrix.toarray()
                if config.edge_type == 'tx-tx':
                    adj_matrix = adj_matrix[np.ix_(selected_gene_indices, selected_gene_indices)]
                elif config.edge_type == 'tx-bd':
                    adj_matrix = adj_matrix[np.ix_(selected_gene_indices, range(len(config.cell_to_idx)))]
                
                visualizer.plot_attention_matrix(
                    adj_matrix, selected_gene_names, selected_gene_names,
                    f'Layer {layer_idx+1} Head {head_idx+1} Attention Pattern', figures_dir
                )

def visualize_all_attention_patterns_with_metrics(attention_gene_matrix_dict, edge_type='tx-tx', gene_to_idx=None):
    """
    Create a comprehensive visualization of attention patterns and their metrics for all layers and heads.
    
    Parameters:
    -----------
    attention_gene_matrix_dict : dict
        Dictionary containing adjacency matrices for each layer and head
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
    save_path = figures_dir / 'dataset_visualization' / f'all_attention_metrics_{edge_type}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig_patterns, fig_metrics 

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
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    @staticmethod
    def create_gene_type_colors(gene_types_dict: Dict[str, str], gene_type_to_color: Optional[Dict[str, str]] = None) -> Dict[str, Tuple[float, float, float]]:
        """Create color mapping for gene types."""
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            # Convert hex colors to RGB tuples
            return {t: GeneTypeVisualizer.hex_to_rgb(gene_type_to_color.get(t, '#808080')) for t in unique_types}
        else:
            # Fallback to default color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            return dict(zip(unique_types, colors))
    
    @staticmethod
    def add_gene_type_coloring(ax, gene_types_dict: Dict[str, str], 
                             type_to_color: Dict[str, Tuple[float, float, float]], axis: str = 'y'):
        """Add gene type coloring to axis labels."""
        labels = ax.get_yticklabels() if axis == 'y' else ax.get_xticklabels()
        gene_types = [gene_types_dict.get(label.get_text(), '') for label in labels]
        
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

class AttentionVisualizer:
    """Main class for visualizing attention patterns."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.processor = AttentionMatrixProcessor()
        self.gene_type_visualizer = GeneTypeVisualizer()
        self.type_to_color = self.gene_type_visualizer.create_gene_type_colors(
            self.config.gene_types_dict,
            self.config.gene_type_to_color
        )
    
    def plot_attention_matrix(self, matrix: np.ndarray, gene_names: List[str], 
                            ordered_gene_names: List[str], title: str, figures_dir: Path):
        """Plot attention matrix with gene ordering."""
        if self.config.edge_type == 'tx-tx':
            self._plot_tx_tx_matrix(matrix, gene_names, ordered_gene_names, title, figures_dir)
        else:
            self._plot_tx_bd_matrix(matrix, gene_names, ordered_gene_names, title, figures_dir)

    def plot_all_layers_heads(self, attention_gene_matrix_dict: Dict, selected_gene_names: List[str], 
                            selected_gene_indices: List[int], figures_dir: Path, cell_order=None):
        """Plot all layers and heads in a single figure grid."""
        n_layers = len(attention_gene_matrix_dict['adj_matrix'])
        n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
        
        # Create a large figure with subplots for each layer-head combination
        fig, axes = plt.subplots(n_heads, n_layers, 
                                figsize=(n_layers * 5, n_heads * 5),
                                constrained_layout=False)
        
        # If there's only one layer or head, axes will be 1D
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        if n_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each layer-head combination
        for head_idx in range(n_heads):
            for layer_idx in range(n_layers):
                ax = axes[head_idx, layer_idx]
                
                # Get the attention matrix for this layer-head combination
                adj_matrix = attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx]
                if isinstance(adj_matrix, lil_matrix):
                    adj_matrix = adj_matrix.toarray()
                
                # Select the appropriate subset of the attention matrix
                if self.config.edge_type == 'tx-tx':
                    adj_matrix = adj_matrix[np.ix_(selected_gene_indices, selected_gene_indices)]
                    df = pd.DataFrame(adj_matrix, index=selected_gene_names, columns=selected_gene_names)
                else:  # tx-bd
                    adj_matrix = adj_matrix[np.ix_(selected_gene_indices, range(len(self.config.cell_to_idx)))]
                    # Use cell IDs as column names
                    cell_ids = list(self.config.cell_to_idx.keys())
                    df = pd.DataFrame(adj_matrix, index=selected_gene_names, columns=cell_ids)
                
                    # Reorder columns based on cell_order if provided
                    if cell_order is not None:
                        # Ensure all cells in cell_order exist in the dataframe
                        valid_cells = [cell for cell in cell_order if cell in df.columns]
                        df = df[valid_cells]
                
                # Plot heatmap
                sns.heatmap(df, cmap=self.config.cmap, ax=ax, cbar=False)
                
                # Add gene type coloring if available
                if self.config.gene_types_dict:
                    self.gene_type_visualizer.add_gene_type_coloring(ax, self.config.gene_types_dict, self.type_to_color, 'y')
                    if self.config.edge_type == 'tx-tx':
                        self.gene_type_visualizer.add_gene_type_coloring(ax, self.config.gene_types_dict, self.type_to_color, 'x')
                
                # Set title and labels
                ax.set_title(f'L{layer_idx+1}H{head_idx+1}')
                
                # Remove all x-axis labels for better readability
                ax.set_xticklabels([])
                ax.set_xlabel('')
                
                # Only show y-axis labels for the leftmost subplot
                if layer_idx > 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout to make room for colorbar and legend
        plt.subplots_adjust(right=0.85, top=0.95, bottom=0.05, left=0.1)
        
        # Add a colorbar to the right of the figure
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=self.config.cmap)
        fig.colorbar(sm, cax=cbar_ax, label=self.config.cbar_label)
        
        # Add gene type legend if available
        if self.config.gene_types_dict:
            self.gene_type_visualizer.add_gene_type_legend(ax, self.type_to_color)
        
        # Save the plot
        save_path = figures_dir / 'dataset_visualization' / f'all_layers_heads_{self.config.edge_type}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

    def _plot_tx_tx_matrix(self, matrix: np.ndarray, gene_names: List[str], 
                          ordered_gene_names: List[str], title: str, figures_dir: Path):
        """Plot transcript-to-transcript attention matrix."""
        df = pd.DataFrame(matrix, index=gene_names, columns=gene_names)
        df = df.reindex(index=ordered_gene_names, columns=ordered_gene_names)
        
        plt.figure(figsize=self.config.figsize)
        ax = sns.heatmap(df, cmap=self.config.cmap, cbar_kws={'label': self.config.cbar_label})
        
        if self.config.gene_types_dict:
            self.gene_type_visualizer.add_gene_type_coloring(ax, self.config.gene_types_dict, self.type_to_color, 'y')
            self.gene_type_visualizer.add_gene_type_coloring(ax, self.config.gene_types_dict, self.type_to_color, 'x')
            self.gene_type_visualizer.add_gene_type_legend(ax, self.type_to_color)
        
        plt.title(title)
        plt.xlabel('Target Gene')
        plt.ylabel('Source Gene')
        plt.tight_layout()
        self._save_plot(figures_dir, title)
    
    def _plot_tx_bd_matrix(self, matrix: np.ndarray, gene_names: List[str], 
                          ordered_gene_names: List[str], title: str, figures_dir: Path):
        """Plot transcript-to-binding attention matrix."""
        if not self.config.cell_to_idx:
            raise ValueError("cell_to_idx must be provided for tx-bd edge type")
        
        # Use cell IDs as column names instead of cell types
        cell_ids = list(self.config.cell_to_idx.keys())
        df = pd.DataFrame(matrix, index=gene_names, columns=cell_ids)
        df = df.reindex(index=ordered_gene_names)
        
        # Reorder columns based on cell_order if provided
        if self.config.cell_order is not None:
            # Ensure all cells in cell_order exist in the dataframe
            valid_cells = [cell for cell in self.config.cell_order if cell in df.columns]
            df = df[valid_cells]
        
        plt.figure(figsize=self.config.figsize)
        ax = sns.heatmap(df, cmap=self.config.cmap, cbar_kws={'label': self.config.cbar_label})
        
        if self.config.gene_types_dict:
            self.gene_type_visualizer.add_gene_type_coloring(ax, self.config.gene_types_dict, self.type_to_color, 'y')
            self.gene_type_visualizer.add_gene_type_legend(ax, self.type_to_color)
        
        plt.title(title)
        plt.xlabel('Target Cell ID')
        plt.ylabel('Source Gene')
        plt.tight_layout()
        self._save_plot(figures_dir, title)
    
    def _save_plot(self, figures_dir: Path, title: str):
        """Save plot to file."""
        save_path = figures_dir / 'dataset_visualization' / f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

def create_figures_dir(edge_type: str) -> Path:
    """Create and return the figures directory."""
    figures_dir = Path('figures') / edge_type
    figures_dir.mkdir(exist_ok=True)
    (figures_dir / 'dataset_visualization').mkdir(exist_ok=True)
    return figures_dir 

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
    gene_type_to_color: Optional[Dict[str, str]] = None

class AttentionMatrixAggregator:
    """Handles aggregation of attention matrices across layers and heads."""
    
    @staticmethod
    def compute_layer_average_attention(attention_gene_matrix_dict: Dict, layer_idx: int, selected_gene_indices: Optional[List[int]] = None, edge_type: str = 'tx-tx') -> np.ndarray:
        """Compute average attention weights across all heads for a given layer."""
        n_heads = len(attention_gene_matrix_dict['adj_matrix'][layer_idx])
        if selected_gene_indices is not None:
            if edge_type == 'tx-tx':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][layer_idx][0].toarray()[selected_gene_indices, :][:, selected_gene_indices])
            elif edge_type == 'tx-bd':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][layer_idx][0].toarray()[selected_gene_indices, :])
        else:
            avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][layer_idx][0].toarray())
        
        for head_idx in range(n_heads):
            if selected_gene_indices is not None:
                if edge_type == 'tx-tx':
                    avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :][:, selected_gene_indices]
                elif edge_type == 'tx-bd':
                    avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :]
            else:
                avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / n_heads
    
    @staticmethod
    def compute_head_average_attention(attention_gene_matrix_dict: Dict, head_idx: int, selected_gene_indices: Optional[List[int]] = None, edge_type: str = 'tx-tx') -> np.ndarray:
        """Compute average attention weights across all layers for a given head."""
        n_layers = len(attention_gene_matrix_dict['adj_matrix'])
        if selected_gene_indices is not None:
            if edge_type == 'tx-tx':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][head_idx].toarray()[selected_gene_indices, :][:, selected_gene_indices])
            elif edge_type == 'tx-bd':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][head_idx].toarray()[selected_gene_indices, :])
        else:
            avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][head_idx].toarray())
        
        for layer_idx in range(n_layers):
            if selected_gene_indices is not None:
                if edge_type == 'tx-tx':
                    avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :][:, selected_gene_indices]
                elif edge_type == 'tx-bd':
                    avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :]
            else:
                avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / n_layers
    
    @staticmethod
    def compute_overall_average_attention(attention_gene_matrix_dict: Dict, selected_gene_indices: Optional[List[int]] = None, edge_type: str = 'tx-tx') -> np.ndarray:
        """Compute average attention weights across all layers and heads."""
        n_layers = len(attention_gene_matrix_dict['adj_matrix'])
        n_heads = len(attention_gene_matrix_dict['adj_matrix'][0])
        if selected_gene_indices is not None:
            if edge_type == 'tx-tx':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray()[selected_gene_indices, :][:, selected_gene_indices])
            elif edge_type == 'tx-bd':
                avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray()[selected_gene_indices, :])
        else:
            avg_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray())
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                if selected_gene_indices is not None:
                    if edge_type == 'tx-tx':
                        avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :][:, selected_gene_indices]
                    elif edge_type == 'tx-bd':
                        avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()[selected_gene_indices, :]
                else:
                    avg_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()
        
        return avg_matrix / (n_layers * n_heads)

class AttentionSummarizer:
    """Handles summarization of attention weights."""
    
    def __init__(self, edge_type: str, gene_to_idx: Optional[Dict[str, int]] = None,
                 cell_to_idx: Optional[Dict[str, int]] = None):
        self.edge_type = edge_type
        self.gene_to_idx = gene_to_idx
        self.cell_to_idx = cell_to_idx
    
    def summarize_attention_by_gene_df(self, attention_df: pd.DataFrame, layer_idx: int, 
                                     head_idx: int) -> Tuple[lil_matrix, lil_matrix]:
        """Summarize attention weights at the gene level."""
        filtered_df = attention_df[
            (attention_df['layer'] == layer_idx + 1) & 
            (attention_df['head'] == head_idx + 1)
        ]
        
        if not self.gene_to_idx:
            raise ValueError("gene_to_idx must be provided")
        
        num_genes = len(self.gene_to_idx)
        
        if self.edge_type == 'tx-tx':
            adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
            count_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
        elif self.edge_type == 'tx-bd':
            if not self.cell_to_idx:
                raise ValueError("cell_to_idx must be provided for tx-bd edge type")
            num_cells = len(self.cell_to_idx)
            adj_matrix = lil_matrix((num_genes, num_cells), dtype=np.float32)
            count_matrix = lil_matrix((num_genes, num_cells), dtype=np.float32)
        else:
            raise ValueError(f"Invalid edge type: {self.edge_type}")
        
        for _, row in filtered_df.iterrows():
            src_idx = self.gene_to_idx[row['source_gene']]
            if self.edge_type == 'tx-tx':
                dst_idx = self.gene_to_idx[row['target_gene']]
            else:  # tx-bd
                dst_idx = self.cell_to_idx[row['target_cell_id']]
            adj_matrix[src_idx, dst_idx] += row['attention_weight']
            count_matrix[src_idx, dst_idx] += 1
        
        return adj_matrix.tolil(), count_matrix.tolil() 