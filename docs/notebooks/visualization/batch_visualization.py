import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_heatmap, downsample_matrix, create_figures_dir

def extract_attention_df(attention_weights, gene_names=None, cell_ids=None, cell_types_dict=None, edge_type='tx-tx'):
    """Extract attention weights into a structured dataset."""
    assert edge_type in ['tx-tx', 'tx-bd'], "Edge type must be 'tx-tx' or 'tx-bd'"
    
    data = []
    for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
        if edge_type == 'tx-tx':
            alpha_tensor = alpha['tx']
            edge_index = edge_index['tx']
        else:
            alpha_tensor = alpha['bd']
            edge_index = edge_index['bd']
        
        alpha_tensor = alpha_tensor.cpu().detach().numpy()
        edge_index = edge_index.cpu().detach().numpy()
        
        for head_idx in range(alpha_tensor.shape[1]):
            head_weights = alpha_tensor[:, head_idx]
            
            for i, (src, dst) in enumerate(edge_index.T):
                entry = {
                    'source': int(src),
                    'target': int(dst),
                    'edge_type': edge_type,
                    'layer': layer_idx + 1,
                    'head': head_idx + 1,
                    'attention_weight': float(head_weights[i])
                }
                
                if gene_names is not None:
                    entry['source_gene'] = gene_names[src]
                    if edge_type == 'tx-tx':
                        entry['target_gene'] = gene_names[dst]
                    else:
                        entry['target_cell'] = cell_types_dict[cell_ids[dst]]
                
                data.append(entry)
    
    return pd.DataFrame(data)

def visualize_attention_df(attention_df, layer_idx, head_idx, edge_type, gene_types_dict=None, cell_types_dict=None, gene_type_to_color=None):
    """
    Visualize attention weights as a heatmap for a specific layer and head.
    
    Parameters:
    -----------
    attention_df : pandas.DataFrame
        DataFrame containing attention weights with columns:
        - 'layer': Layer index (1-indexed)
        - 'head': Head index (1-indexed)
        - 'source': Source node index
        - 'target': Target node index
        - 'attention_weight': Attention weight
        - 'source_gene': Source gene name (if gene_names provided)
        - 'target_gene'/'target_cell': Target gene/cell type (if gene_names provided)
    layer_idx : int
        Layer index (1-indexed)
    head_idx : int
        Head index (1-indexed)
    edge_type : str
        Edge type ("tx-tx" or "tx-bd")
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    cell_types_dict : dict, optional
        Dictionary mapping cell ids to their types
    gene_type_to_color : dict, optional
        Dictionary mapping gene types to colors
    """
    assert edge_type in ["tx-tx", "tx-bd"], "edge_type must be 'tx-tx' or 'tx-bd'"
    
    # Filter data based on layer and head
    filtered_df = attention_df[
        (attention_df['layer'] == layer_idx + 1) & 
        (attention_df['head'] == head_idx + 1)
    ]
    
    if filtered_df.empty:
        print(f"No data found for layer {layer_idx + 1} and head {head_idx + 1}")
        return
    
    # Determine source and target columns based on edge type
    if edge_type == "tx-tx":
        source_col = "source_gene"
        target_col = "target_gene"
        if gene_types_dict is not None:
            filtered_df = filtered_df[filtered_df[source_col].isin(gene_types_dict.keys())]
            filtered_df = filtered_df[filtered_df[target_col].isin(gene_types_dict.keys())]
    else:  # tx-bd
        source_col = "source_gene"
        target_col = "target_cell"
        if cell_types_dict is not None:
            filtered_df = filtered_df[filtered_df[source_col].isin(gene_types_dict.keys())]
    
    # Get unique source and target nodes
    unique_sources = sorted(filtered_df[source_col].unique())
    unique_targets = sorted(filtered_df[target_col].unique())
    
    # Initialize matrices for averaging
    num_source_nodes = len(unique_sources)
    num_target_nodes = len(unique_targets)
    total_attention = np.zeros((num_source_nodes, num_target_nodes))
    count_matrix = np.zeros((num_source_nodes, num_target_nodes))
    
    # Create mapping from node indices to matrix indices
    source_to_idx = {node: idx for idx, node in enumerate(unique_sources)}
    target_to_idx = {node: idx for idx, node in enumerate(unique_targets)}
    
    # Compute average attention weights
    for _, row in filtered_df.iterrows():
        src = row[source_col]
        dst = row[target_col]
        src_idx = source_to_idx[src]
        dst_idx = target_to_idx[dst]
        total_attention[src_idx, dst_idx] += row['attention_weight']
        count_matrix[src_idx, dst_idx] += 1
    
    # Compute average attention matrix
    adj_matrix = np.divide(total_attention, count_matrix, 
                          out=np.zeros_like(total_attention), 
                          where=count_matrix!=0)
    
    # Sort by gene types if provided
    sorted_sources = unique_sources
    sorted_targets = unique_targets
    if gene_types_dict is not None:
        source_types = [gene_types_dict.get(src, '') for src in unique_sources]
        sorted_indices = sorted(range(len(unique_sources)), key=lambda i: source_types[i])
        sorted_sources = [unique_sources[i] for i in sorted_indices]
        if edge_type == 'tx-tx':
            sorted_targets = sorted_sources
            adj_matrix = adj_matrix[sorted_indices][:, sorted_indices]
        else:
            adj_matrix = adj_matrix[sorted_indices]
    
    # Create and save figure
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'batch_visualization' / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png'
    
    # Create appropriate title
    title = f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}'
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111)
    
    # Create main heatmap
    sns.heatmap(adj_matrix, ax=ax, cmap='viridis', xticklabels=sorted_targets, yticklabels=sorted_sources)
    
    if gene_types_dict is not None:
        # Create color mapping for gene types
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            # Use provided colors for gene types
            type_to_color = {t: gene_type_to_color.get(t, plt.cm.Set3(0)) for t in unique_types}
        else:
            # Fallback to default color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
        
        # Get gene types for source and target
        source_types = [gene_types_dict.get(gene, '') for gene in sorted_sources]
        if edge_type == 'tx-tx':
            target_types = source_types
        else:
            target_types = [cell_types_dict.get(cell, '') for cell in sorted_targets]
        
        # Color the y-axis labels (source genes)
        y_labels = ax.get_yticklabels()
        for label, gene_type in zip(y_labels, source_types):
            color = type_to_color[gene_type]
            # Convert RGB to RGBA with alpha for highlight
            highlight_color = (*color[:3], 0.3)  # 30% opacity
            label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
        
        # Color the x-axis labels (target genes/cells)
        x_labels = ax.get_xticklabels()
        if edge_type == 'tx-tx':
            for label, gene_type in zip(x_labels, target_types):
                color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))  # gray for unknown types
                # Convert RGB to RGBA with alpha for highlight
                highlight_color = (*color[:3], 0.3)  # 30% opacity
                label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
        
        # Create legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=gene_type)
                          for gene_type, color in type_to_color.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if edge_type == 'tx-tx':
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Source Gene')
    else:
        ax.set_xlabel('Target Cell')
        ax.set_ylabel('Source Gene')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_mean_attention_df(attention_df, layer_idx=None, head_idx=None, edge_type='tx-tx', 
                              gene_types_dict=None, cell_types_dict=None, gene_type_to_color=None):
    """
    Visualize mean attention weights as a heatmap, averaging across layers and/or heads.
    
    Parameters:
    -----------
    attention_df : pandas.DataFrame
        DataFrame containing attention weights
    layer_idx : int or None, optional
        If provided, average across this specific layer. If None, average across all layers.
    head_idx : int or None, optional
        If provided, average across this specific head. If None, average across all heads.
    edge_type : str, optional
        Edge type ("tx-tx" or "tx-bd")
    gene_types_dict : dict, optional
        Dictionary mapping gene names to their types
    cell_types_dict : dict, optional
        Dictionary mapping cell ids to their types
    gene_type_to_color : dict, optional
        Dictionary mapping gene types to colors
    """
    assert edge_type in ["tx-tx", "tx-bd"], "edge_type must be 'tx-tx' or 'tx-bd'"
    
    # Filter data based on layer and head if specified
    filtered_df = attention_df.copy()
    if layer_idx is not None:
        filtered_df = filtered_df[filtered_df['layer'] == layer_idx + 1]
    if head_idx is not None:
        filtered_df = filtered_df[filtered_df['head'] == head_idx + 1]
    
    if filtered_df.empty:
        print(f"No data found for the specified parameters")
        return
    
    # Determine source and target columns based on edge type
    if edge_type == "tx-tx":
        source_col = "source_gene"
        target_col = "target_gene"
        if gene_types_dict is not None:
            filtered_df = filtered_df[filtered_df[source_col].isin(gene_types_dict.keys())]
            filtered_df = filtered_df[filtered_df[target_col].isin(gene_types_dict.keys())]
    else:  # tx-bd
        source_col = "source_gene"
        target_col = "target_cell"
        if cell_types_dict is not None:
            filtered_df = filtered_df[filtered_df[source_col].isin(gene_types_dict.keys())]
    
    # Get unique source and target nodes
    unique_sources = sorted(filtered_df[source_col].unique())
    unique_targets = sorted(filtered_df[target_col].unique())
    
    # Initialize matrices for averaging
    num_source_nodes = len(unique_sources)
    num_target_nodes = len(unique_targets)
    total_attention = np.zeros((num_source_nodes, num_target_nodes))
    count_matrix = np.zeros((num_source_nodes, num_target_nodes))
    
    # Create mapping from node indices to matrix indices
    source_to_idx = {node: idx for idx, node in enumerate(unique_sources)}
    target_to_idx = {node: idx for idx, node in enumerate(unique_targets)}
    
    # Compute average attention weights
    for _, row in filtered_df.iterrows():
        src = row[source_col]
        dst = row[target_col]
        src_idx = source_to_idx[src]
        dst_idx = target_to_idx[dst]
        total_attention[src_idx, dst_idx] += row['attention_weight']
        count_matrix[src_idx, dst_idx] += 1
    
    # Compute average attention matrix
    adj_matrix = np.divide(total_attention, count_matrix, 
                          out=np.zeros_like(total_attention), 
                          where=count_matrix!=0)
    
    # Sort by gene types if provided
    sorted_sources = unique_sources
    sorted_targets = unique_targets
    if gene_types_dict is not None:
        source_types = [gene_types_dict.get(src, '') for src in unique_sources]
        sorted_indices = sorted(range(len(unique_sources)), key=lambda i: source_types[i])
        sorted_sources = [unique_sources[i] for i in sorted_indices]
        if edge_type == 'tx-tx':
            sorted_targets = sorted_sources
            adj_matrix = adj_matrix[sorted_indices][:, sorted_indices]
        else:
            adj_matrix = adj_matrix[sorted_indices]
    
    # Create and save figure
    figures_dir = create_figures_dir()
    
    # Handle filename for mean cases
    layer_str = f"layer_{layer_idx + 1}" if layer_idx is not None else "all_layers"
    head_str = f"head_{head_idx + 1}" if head_idx is not None else "all_heads"
    
    save_path = figures_dir / f'batch_visualization' / f'attention_{layer_str}_{head_str}_{edge_type}.png'
    
    # Create appropriate title
    title = f'Mean Attention Weights - {layer_str}, {head_str}, Edge Type: {edge_type}'
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111)
    
    # Create main heatmap
    sns.heatmap(adj_matrix, ax=ax, cmap='viridis', xticklabels=sorted_targets, yticklabels=sorted_sources)
    
    if gene_types_dict is not None:
        # Create color mapping for gene types
        unique_types = sorted(set(gene_types_dict.values()))
        if gene_type_to_color is not None:
            # Use provided colors for gene types
            type_to_color = {t: gene_type_to_color.get(t, plt.cm.Set3(0)) for t in unique_types}
        else:
            # Fallback to default color palette
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            type_to_color = dict(zip(unique_types, colors))
        
        # Get gene types for source and target
        source_types = [gene_types_dict.get(gene, '') for gene in sorted_sources]
        if edge_type == 'tx-tx':
            target_types = source_types
        else:
            target_types = [cell_types_dict.get(cell, '') for cell in sorted_targets]
        
        # Color the y-axis labels (source genes)
        y_labels = ax.get_yticklabels()
        for label, gene_type in zip(y_labels, source_types):
            color = type_to_color[gene_type]
            # Convert RGB to RGBA with alpha for highlight
            highlight_color = (*color[:3], 0.3)  # 30% opacity
            label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
        
        # Color the x-axis labels (target genes/cells)
        x_labels = ax.get_xticklabels()
        if edge_type == 'tx-tx':
            for label, gene_type in zip(x_labels, target_types):
                color = type_to_color.get(gene_type, (0.5, 0.5, 0.5))  # gray for unknown types
                # Convert RGB to RGBA with alpha for highlight
                highlight_color = (*color[:3], 0.3)  # 30% opacity
                label.set_bbox(dict(facecolor=highlight_color, edgecolor='none', alpha=0.3))
        
        # Create legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=gene_type)
                          for gene_type, color in type_to_color.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if edge_type == 'tx-tx':
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Source Gene')
    else:
        ax.set_xlabel('Target Cell')
        ax.set_ylabel('Source Gene')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()