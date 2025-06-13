import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.data import HeteroData
from typing import Optional, Tuple, Dict, List
import pandas as pd
from .utils import hex_to_rgb

def visualize_tile_with_attention_interactive(
    batch: HeteroData,
    attention_weights: Optional[Dict[str, torch.Tensor]] = None,
    edge_types: List[str] = ["tx-bd", "tx-tx"],
    save_path: Optional[str] = None,
    idx_to_cell: Optional[Dict[int, str]] = None,
    idx_to_gene: Optional[Dict[int, str]] = None,
    gene_types_dict: Optional[Dict[str, str]] = None,
    cell_types_dict: Optional[Dict[str, str]] = None,
    cell_type_to_color: Optional[Dict[str, str]] = None,
    gene_type_to_color: Optional[Dict[str, str]] = None,
    boundaries_file: Optional[str] = None,
    nucleus_file: Optional[str] = None,
    transcript_size: int = 20,
    min_linewidth: float = 0.5,
    max_linewidth: float = 3.0,
    attention_threshold: float = 0.0
) -> None:
    """
    Interactive visualization of a tile showing cells (boundaries), transcripts, and their attention-weighted connections.
    The attention weights are represented by the thickness of the connecting lines.
    
    Parameters
    ----------
    batch : HeteroData
        PyTorch Geometric HeteroData object containing the tile data
    attention_weights : Optional[Dict[str, torch.Tensor]]
        Dictionary containing attention weights for different edge types
    edge_types : List[str]
        List of edge types to visualize (default: ["tx-bd", "tx-tx"])
    save_path : Optional[str]
        If provided, save the figure to this path
    idx_to_cell : Optional[Dict[int, str]]
        Dictionary mapping indices to cell ids
    idx_to_gene : Optional[Dict[int, str]]
        Dictionary mapping transcript IDs to gene names
    gene_types_dict : Optional[Dict[str, str]]
        Dictionary mapping gene names to their types
    cell_types_dict : Optional[Dict[str, str]]
        Dictionary mapping cell ids to their types
    cell_type_to_color : Optional[Dict[str, str]]
        Dictionary mapping cell types to colors
    gene_type_to_color : Optional[Dict[str, str]]
        Dictionary mapping gene types to colors
    boundaries_file : Optional[str]
        Path to the cell boundaries parquet file
    nucleus_file : Optional[str]
        Path to the nuclei parquet file
    transcript_size : int
        Size of transcript points in the plot
    min_linewidth : float
        Minimum line width for edges
    max_linewidth : float
        Maximum line width for edges
    attention_threshold : float
        Minimum attention weight threshold for displaying edges
    """
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
    
    # Create figure
    fig = go.Figure()
    
    # Create color maps
    if gene_types_dict is not None:
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            type_to_color = {t: gene_type_to_color.get(t, '#808080') for t in unique_types}
        else:
            # Fallback to default color palette
            colors = ['#%02x%02x%02x' % tuple(np.random.randint(0, 255, 3)) for _ in range(len(unique_types))]
            type_to_color = dict(zip(unique_types, colors))
        
        gene_to_type = {gene: gene_types_dict.get(gene, '') for gene in unique_genes}
    else:
        n_genes = len(unique_genes)
        colors = ['#%02x%02x%02x' % tuple(np.random.randint(0, 255, 3)) for _ in range(n_genes)]
        gene_color_map = dict(zip(unique_genes, colors))
    
    # Plot nuclei
    if nucleus_file is not None:
        nucleus = pd.read_parquet(nucleus_file)
        for cell_id, cell_data in nucleus.groupby('cell_id'):
            x_coords = cell_data['vertex_x'].values
            y_coords = cell_data['vertex_y'].values
            
            # Close the polygon
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
            
            # Add nucleus trace
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='red', width=1),
                name='Nucleus',
                showlegend=True if cell_id == nucleus['cell_id'].iloc[0] else False
            ))
            
            # Add cell type label
            if cell_id in cell_ids and cell_types_dict is not None:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                cell_type = cell_types_dict[cell_id]
                cell_color = cell_type_to_color.get(cell_type, 'red') if cell_type_to_color else 'red'
                fig.add_annotation(
                    x=center_x,
                    y=center_y,
                    text=str(cell_type),
                    showarrow=False,
                    font=dict(size=8, color=cell_color),
                    bgcolor='white',
                    bordercolor='white',
                    borderwidth=0,
                    opacity=0.7
                )
    
    # Plot cell boundaries
    if boundaries_file is not None:
        df = pd.read_parquet(boundaries_file)
        for cell_id, cell_data in df.groupby('cell_id'):
            x_coords = cell_data['vertex_x'].values
            y_coords = cell_data['vertex_y'].values
            
            # Close the polygon
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='gray', width=1),
                name='Cell Boundary',
                showlegend=True if cell_id == df['cell_id'].iloc[0] else False
            ))
    
    # Plot transcripts
    if gene_types_dict is not None:
        for gene_type in unique_types:
            type_mask = [gene_to_type.get(gene, '') == gene_type for gene in gene_names]
            if np.any(type_mask):
                fig.add_trace(go.Scatter(
                    x=tx_pos[type_mask, 0],
                    y=tx_pos[type_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=transcript_size,
                        color=type_to_color[gene_type],
                        opacity=0.6
                    ),
                    name=gene_type
                ))
    else:
        for gene_name in unique_genes:
            gene_mask = gene_names == gene_name
            if np.any(gene_mask):
                fig.add_trace(go.Scatter(
                    x=tx_pos[gene_mask, 0],
                    y=tx_pos[gene_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=transcript_size,
                        color=gene_color_map[gene_name],
                        opacity=0.6
                    ),
                    name=gene_name
                ))
    
    # Plot attention-weighted edges
    if attention_weights is not None and idx_to_cell is not None:
        for edge_type in edge_types:
            if edge_type in attention_weights:
                if edge_type == "tx-tx":
                    edge_index = batch[("tx", "neighbors", "tx")].edge_index.cpu().numpy()
                    attn_weights = attention_weights[edge_type].cpu().numpy()
                else:  # tx-bd
                    edge_index = batch[("tx", "belongs", "bd")].edge_index.cpu().numpy()
                    attn_weights = attention_weights[edge_type].cpu().numpy()
                
                # Filter edges by attention threshold
                mask = attn_weights >= attention_threshold
                edge_index = edge_index[:, mask]
                attn_weights = attn_weights[mask]
                
                if len(attn_weights) > 0:
                    # Normalize attention weights for line width
                    norm_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
                    linewidths = min_linewidth + (max_linewidth - min_linewidth) * norm_weights
                    
                    # Set colors based on edge type
                    edge_color = 'green' if edge_type == "tx-bd" else 'purple'
                    edge_label = 'Transcript-Cell' if edge_type == "tx-bd" else 'Transcript-Transcript'
                    
                    # Plot edges
                    for i in range(edge_index.shape[1]):
                        if edge_type == "tx-bd":
                            tx_idx, bd_idx = edge_index[:, i]
                            tx_pos_i = tx_pos[tx_idx]
                            bd_pos_i = bd_pos[bd_idx]
                        else:  # tx-tx
                            tx1_idx, tx2_idx = edge_index[:, i]
                            tx_pos_i = tx_pos[tx1_idx]
                            bd_pos_i = tx_pos[tx2_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[tx_pos_i[0], bd_pos_i[0]],
                            y=[tx_pos_i[1], bd_pos_i[1]],
                            mode='lines',
                            line=dict(
                                color=edge_color,
                                width=linewidths[i],
                                opacity=0.6
                            ),
                            name=edge_label,
                            showlegend=True if i == 0 else False
                        ))
    
    # Update layout
    fig.update_layout(
        title='Interactive Tile Visualization with Attention Weights',
        xaxis_title='X Coordinate (µm)',
        yaxis_title='Y Coordinate (µm)',
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Save figure if path is provided
    if save_path:
        fig.write_html(save_path)
    
    return fig