import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import streamlit as st
from utils import load_attention_data, get_top_genes_by_attention
from attention_data_processing import save_intermediate_data, load_intermediate_data

def create_interactive_attention_plot(
    attention_gene_matrix_dict: Dict,
    layer_idx: int | str,
    head_idx: int | str,
    top_genes: List[str],
    top_indices: np.ndarray,
    layer_avg_matrices: np.ndarray,
    head_avg_matrices: np.ndarray,
    average_attention_matrix: np.ndarray,
    top_k: int = 20,
    threshold: float = 0.0
) -> go.Figure:
    """Create an interactive plot showing attention weights between genes.
    
    Args:
        attention_gene_matrix_dict: Dictionary containing attention matrices
        layer_idx: Index of the layer to visualize or 'average' for average across layers
        head_idx: Index of the attention head to visualize or 'average' for average across heads
        top_genes: List of top gene names
        top_indices: Indices of top genes
        layer_avg_matrices: Pre-computed layer-wise average attention matrices
        head_avg_matrices: Pre-computed head-wise average attention matrices
        average_attention_matrix: Pre-computed overall average attention matrix
        top_k: Number of top genes to show
        threshold: Minimum attention weight threshold (0-1) for showing edges
    """
    # Get the appropriate attention matrix based on layer and head selection
    if layer_idx == 'average' and head_idx == 'average':
        top_matrix = average_attention_matrix[top_indices][:, top_indices]
    elif layer_idx == 'average':
        top_matrix = head_avg_matrices[head_idx][top_indices][:, top_indices]
    elif head_idx == 'average':
        top_matrix = layer_avg_matrices[layer_idx][top_indices][:, top_indices]
    else:
        # Get attention matrix for specified layer and head
        attention_matrix = attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx]
        top_matrix = attention_matrix[top_indices][:, top_indices].toarray()
    
    # Create node positions (two columns)
    n_genes = len(top_genes)
    x_pos = np.array([0] * n_genes + [1] * n_genes)
    y_pos = np.array(list(range(n_genes)) + list(range(n_genes)))
    
    # Create figure
    fig = go.Figure()
    
    # Collect all edge data
    edge_data = []
    for i in range(n_genes):
        for j in range(n_genes):
            weight = float(top_matrix[i, j])
            if weight > threshold:  # Only show edges above threshold
                edge_data.append({
                    'x': [0, 1],
                    'y': [i, j],
                    'weight': weight
                })
    
    # Sort edge data by weight
    edge_data.sort(key=lambda x: x['weight'])
    
    # Create color ranges
    colors = ['#440154', '#482475', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    weight_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    # Add edges for each weight range
    for color, (min_weight, max_weight) in zip(colors, weight_ranges):
        filtered_edges = [edge for edge in edge_data if min_weight <= edge['weight'] < max_weight]
        
        if filtered_edges:
            x_coords = []
            y_coords = []
            weights = []
            
            for edge in filtered_edges:
                x_coords.extend(edge['x'] + [None])
                y_coords.extend(edge['y'] + [None])
                weights.extend([edge['weight'], edge['weight'], None])
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    width=2,
                    color=color
                ),
                hoverinfo='text',
                hovertext=[f'Attention weight: {w:.3f}' if w is not None else None for w in weights],
                name=f'{min_weight:.1f} - {max_weight:.1f}',
                showlegend=True
            ))
    
    # Add nodes (genes)
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=top_genes * 2,  # Duplicate gene names for both columns
        textposition="middle center",
        hoverinfo='text',
        hovertext=[f'Gene: {gene}' for gene in top_genes * 2],
        name='Genes',
        showlegend=True
    ))
    
    # Update layout
    layer_title = "Average" if layer_idx == 'average' else f"Layer {layer_idx+1}"
    head_title = "Average" if head_idx == 'average' else f"Head {head_idx+1}"
    
    fig.update_layout(
        title=f'Interactive Gene Attention Weights ({layer_title}, {head_title})',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        hovermode='closest',
        width=800,
        height=600,
        legend=dict(
            title="Attention Weight Ranges",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def main():
    st.title("Interactive Gene Attention Visualization")
    
    # Create intermediate data directory
    intermediate_dir = Path('intermediate_data')
    
    # Check if intermediate data exists
    if not intermediate_dir.exists() or not list(intermediate_dir.glob('*.pkl')):
        st.info("First time loading: Computing and saving intermediate data...")
        # Load data
        attention_gene_matrix_dict = load_attention_data()
        
        # Load gene names
        transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')
        gene_names = transcripts['feature_name'].unique().tolist()
        
        # Save intermediate data
        save_intermediate_data(attention_gene_matrix_dict, gene_names, intermediate_dir)
        st.success("Intermediate data saved successfully!")

    # Load intermediate data
    attention_gene_matrix_dict, gene_names, average_attention_matrix, layer_avg_matrices, head_avg_matrices, top_genes_dict = load_intermediate_data(intermediate_dir)
    
    # print(top_genes_dict) # for debugging; also print out the top_k genes and names
    
    # Get dimensions
    n_layers = len(attention_gene_matrix_dict["adj_matrix"])
    n_heads = len(attention_gene_matrix_dict["adj_matrix"][0])
    
    # Create sidebar controls
    st.sidebar.header("Visualization Parameters")
    top_k = st.sidebar.slider(
        "Number of Top Genes",
        min_value=5,
        max_value=50,
        value=20
    )
    
    threshold = st.sidebar.slider(
        "Attention Weight Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Only show attention weights above this threshold"
    )
    
    # Get pre-computed top genes
    top_genes, top_indices = top_genes_dict[50]
    
    # reverse the order of the top genes and indices and then take the top k genes
    top_genes = top_genes[::-1][:top_k]
    top_indices = top_indices[::-1][:top_k]
    
    # Create layer selection with average option
    layer_options = ['average'] + list(range(n_layers))
    layer_idx = st.sidebar.selectbox(
        "Select Layer",
        options=layer_options,
        format_func=lambda x: "Average" if x == 'average' else f"Layer {x+1}"
    )
    
    # Create head selection with average option
    head_options = ['average'] + list(range(n_heads))
    head_idx = st.sidebar.selectbox(
        "Select Head",
        options=head_options,
        format_func=lambda x: "Average" if x == 'average' else f"Head {x+1}"
    )
    
    # Create and display the plot
    fig = create_interactive_attention_plot(
        attention_gene_matrix_dict,
        layer_idx,
        head_idx,
        top_genes,
        top_indices,
        layer_avg_matrices,
        head_avg_matrices,
        average_attention_matrix,
        top_k,
        threshold
    )
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main() 