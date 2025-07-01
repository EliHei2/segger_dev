import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import HeteroData
from typing import Optional, Tuple, Dict, List
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.colors import ListedColormap
from .utils import hex_to_rgb

def visualize_tile_with_attention(
    batch: HeteroData,
    range_x: Tuple[int, int],
    range_y: Tuple[int, int],
    attention_weights: Optional[Dict[str, torch.Tensor]] = None,
    attention_threshold: float = 0.0,
    edge_types: List[str] = ["tx-bd", "tx-tx"],
    save_path: Optional[str] = None,
    idx_to_cell: Optional[Dict[int, str]] = None,
    idx_to_gene: Optional[Dict[int, str]] = None,
    gene_types_dict: Optional[Dict[str, str]] = None,
    cell_types_dict: Optional[Dict[str, str]] = None,
    cell_type_to_color: Optional[Dict[str, str]] = None,
    gene_type_to_color: Optional[Dict[str, str]] = None,
    cell_boundaries: bool = False,
    nucleus_boundaries: bool = False,
    figsize: Tuple[int, int] = (15, 10),
    alpha: float = 0.6,
    transcript_size: int = 20,
    min_linewidth: float = 0.5,
    max_linewidth: float = 3.0
) -> None:
    """
    Visualize a tile showing cells (boundaries), transcripts, and their attention-weighted connections.
    The attention weights are represented by the thickness of the connecting lines.
    
    Parameters
    ----------
    batch : HeteroData
        PyTorch Geometric HeteroData object containing the tile data
    range_x : Tuple[int, int]
        Range of x coordinates to visualize
    range_y : Tuple[int, int]
        Range of y coordinates to visualize
    attention_weights : Optional[Dict[str, torch.Tensor]]
        Dictionary containing attention weights for different edge types
    attention_threshold : float
        Minimum attention weight threshold for displaying edges
    edge_types : List[str]
        List of edge types to visualize (default: ["tx-bd", "tx-tx"])
    save_path : Optional[str]
        If provided, save the figure to this path
    idx_to_cell : Optional[Dict[int, str]]
        Dictionary mapping indices to cell ids. If provided, will plot cell ids on the plot
    idx_to_gene : Optional[Dict[int, str]]
        Dictionary mapping transcript IDs to gene names. If provided, will use gene names in the legend
    gene_types_dict : Optional[Dict[str, str]]
        Dictionary mapping gene names to their types. If provided, will color genes by type
    cell_types_dict : Optional[Dict[str, str]]
        Dictionary mapping cell ids to their types. If provided, will color cells by type
    cell_type_to_color : Optional[Dict[str, str]]
        Dictionary mapping cell types to colors. If provided, will color cells by type
    gene_type_to_color : Optional[Dict[str, str]]
        Dictionary mapping gene types to colors. If provided, will color genes by type
    cell_boundaries : Optional[str]
        Path to the cell boundaries parquet file. If provided, will plot actual cell boundaries
    nucleus_boundaries : Optional[str]
        Path to the nuclei parquet file. If provided, will plot the nuclei on the plot
    figsize : Tuple[int, int]
        Figure size in inches (width, height)
    alpha : float
        Transparency of the attention-weighted edges
    transcript_size : int
        Size of transcript points in the plot
    min_linewidth : float
        Minimum line width for edges (default: 0.5)
    max_linewidth : float
        Maximum line width for edges (default: 3.0)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get positions
    tx_pos = batch["tx"].pos.cpu().numpy()
    bd_pos = batch["bd"].pos.cpu().numpy()
    
    # Get transcript IDs and derive gene names
    transcript_ids = batch["tx"].id.cpu().numpy()
    if idx_to_gene is not None:
        gene_names = np.array([idx_to_gene[tx_id] for tx_id in transcript_ids])
        unique_genes = np.unique(gene_names)
    else:
        gene_names = np.array([f'Transcript {tx_id}' for tx_id in transcript_ids])
        unique_genes = np.unique(gene_names)
    
    # Get cell IDs from PyG data
    cell_ids = batch["bd"].id
    
    # Create color maps
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
        
        # Map genes to their types
        gene_to_type = {gene: gene_types_dict.get(gene, '') for gene in unique_genes}
    else:
        # Create a color map for genes
        n_genes = len(unique_genes)
        gene_colors = plt.cm.tab20(np.linspace(0, 1, n_genes))
        gene_color_map = dict(zip(unique_genes, gene_colors))
    
    # filter out transcripts outside the range
    mask = (tx_pos[:, 0] >= range_x[0]) & (tx_pos[:, 0] <= range_x[1]) & \
           (tx_pos[:, 1] >= range_y[0]) & (tx_pos[:, 1] <= range_y[1])
    
    if gene_types_dict is not None:
        # filter out transcripts with gene_name in gene_restriction
        gene_restriction = [gene for gene in gene_types_dict.keys()]
        mask = mask & np.isin(gene_names, gene_restriction)
    
    # Get available transcripts within range
    tx_indices = np.where(mask)[0]
    tx_pos_filtered = tx_pos[tx_indices]
    gene_names_filtered = gene_names[tx_indices]
    
    # filter out cell boundaries outside the range
    bd_mask = (bd_pos[:, 0] >= range_x[0]) & (bd_pos[:, 0] <= range_x[1]) & \
              (bd_pos[:, 1] >= range_y[0]) & (bd_pos[:, 1] <= range_y[1])
    bd_indices = np.where(bd_mask)[0]
    bd_pos_filtered = bd_pos[bd_mask]
    cell_ids_filtered = cell_ids[bd_mask]
    
    # Plot the nuclei if nucleus_boundaries is True
    if nucleus_boundaries:
        nucleus = pd.read_parquet(Path('data_xenium') / 'nucleus_boundaries.parquet')
        
        # filter out nucleus with cell_id in cell_ids_filtered
        nucleus_filtered = nucleus[nucleus['cell_id'].isin(cell_ids_filtered)]
        
        # plot each cell's nucleus
        for cell_id, cell_data in nucleus_filtered.groupby('cell_id'):
            x_coords = cell_data['vertex_x'].values
            y_coords = cell_data['vertex_y'].values
            
            # Close the polygon
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
            
            # Plot the nucleus
            ax.plot(x_coords, y_coords, color='red', linewidth=1, alpha=0.8,
                    label='Nucleus' if cell_id == nucleus_filtered['cell_id'].iloc[0] else "")
            
            # Add cell ID label at the center of the cell
            if cell_id in cell_ids_filtered and cell_types_dict is not None:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                cell_type = cell_types_dict[cell_id]
                cell_color = cell_type_to_color.get(cell_type, 'red') if cell_type_to_color else 'red'
                ax.text(center_x, center_y, str(cell_type), 
                       ha='center', va='center', fontsize=8, color=cell_color,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        ax.scatter(
            bd_pos_filtered[:, 0],
            bd_pos_filtered[:, 1],
            c='red',
            s=transcript_size*2,
            label='Nuclei',
            alpha=0.8
        )
    
    # Plot cell boundaries if cell_boundaries is True
    if cell_boundaries:
        df = pd.read_parquet(Path('data_xenium') / 'cell_boundaries.parquet')
        
        # Filter boundaries with the cell ids filtered
        df = df[df['cell_id'].isin(cell_ids_filtered)]
        
        # Plot each cell's boundary
        for cell_id, cell_data in df.groupby('cell_id'):
            x_coords = cell_data['vertex_x'].values
            y_coords = cell_data['vertex_y'].values
            
            # Close the polygon
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
            
            # Plot the boundary
            ax.plot(x_coords, y_coords, color='gray', linewidth=1, alpha=0.6,
                    label='Cell Boundary' if cell_id == df['cell_id'].iloc[0] else "")
    
    # Plot transcripts with gene-specific colors
    if gene_types_dict is not None:
        # Plot by gene type
        for gene_type in unique_types:
            type_mask = [gene_to_type.get(gene, '') == gene_type for gene in gene_names_filtered]
            if np.any(type_mask):
                # Plot transcripts
                scatter = ax.scatter(
                    tx_pos_filtered[type_mask, 0],
                    tx_pos_filtered[type_mask, 1],
                    c=[type_to_color[gene_type]],
                    s=transcript_size,
                    label=gene_type,
                    alpha=0.6
                )
                
    else:
        # Plot by gene
        for gene_name in unique_genes:
            gene_mask = gene_names_filtered == gene_name
            if np.any(gene_mask):
                # Plot transcripts
                scatter = ax.scatter(
                    tx_pos_filtered[gene_mask, 0],
                    tx_pos_filtered[gene_mask, 1],
                    c=[gene_color_map[gene_name]],
                    s=transcript_size,
                    label=gene_name,
                    alpha=0.6
                )
    
    # Plot attention-weighted edges if available
    if attention_weights is not None and idx_to_cell is not None:
        for edge_type in edge_types:
            if edge_type in attention_weights:
                if edge_type == "tx-tx":
                    # Get edge indices
                    edge_index = batch[("tx", "neighbors", "tx")].edge_index.cpu().numpy()
                    
                    # Create masks for edges where both nodes are in tx_indices
                    mask = np.isin(edge_index[0], tx_indices) & np.isin(edge_index[1], tx_indices)
                    edge_index = edge_index[:, mask]
                    
                    # Get attention weights for filtered edges
                    attn_weights = attention_weights[edge_type].cpu().numpy()[mask]
                    
                    # Apply attention threshold
                    threshold_mask = attn_weights >= attention_threshold
                    edge_index = edge_index[:, threshold_mask]
                    attn_weights = attn_weights[threshold_mask]
                
                else:  # edge_type == "tx-bd"
                    edge_index = batch[("tx", "belongs", "bd")].edge_index.cpu().numpy()
                    
                    # Create masks for edges where transcript is in tx_indices and boundary is in bd_indices
                    mask = np.isin(edge_index[0], tx_indices) & np.isin(edge_index[1], bd_indices)
                    edge_index = edge_index[:, mask]
                    
                    # Get attention weights for filtered edges
                    attn_weights = attention_weights[edge_type].cpu().numpy()[mask]
                    
                    # Apply attention threshold
                    threshold_mask = attn_weights >= attention_threshold
                    edge_index = edge_index[:, threshold_mask]
                    attn_weights = attn_weights[threshold_mask]
                
                # Normalize attention weights for line width
                if len(attn_weights) > 0:  # Only normalize if we have edges
                    norm_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
                    linewidths = min_linewidth + (max_linewidth - min_linewidth) * norm_weights
                else:
                    continue  # Skip if no edges to plot
                
                # Set colors based on edge type
                if edge_type == "tx-bd":
                    edge_color = 'green'
                    label = 'Transcript-Cell'
                else:  # tx-tx
                    edge_color = 'purple'
                    label = 'Transcript-Transcript'
                
                # Plot edges with attention weights
                for i in range(edge_index.shape[1]):
                    if edge_type == "tx-bd":
                        tx_idx, bd_idx = edge_index[:, i]
                        tx_pos_i = tx_pos[tx_idx]
                        bd_pos_i = bd_pos[bd_idx]
                    else:  # tx-tx
                        tx1_idx, tx2_idx = edge_index[:, i]
                        tx_pos_i = tx_pos[tx1_idx]
                        bd_pos_i = tx_pos[tx2_idx]
                    
                    # Plot edge with thickness based on attention weight
                    ax.plot(
                        [tx_pos_i[0], bd_pos_i[0]],
                        [tx_pos_i[1], bd_pos_i[1]],
                        c=edge_color,
                        alpha=alpha,
                        linewidth=linewidths[i],
                        label=label if i == 0 else None  # Add label only once
                    )
    
    # Customize plot
    ax.set_title('Tile Visualization with Attention Weights')
    ax.set_xlabel('X Coordinate (µm)')
    ax.set_ylabel('Y Coordinate (µm)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()