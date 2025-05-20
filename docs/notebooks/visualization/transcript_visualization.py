import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_heatmap, downsample_matrix, create_figures_dir

def extract_attention_df(attention_weights, gene_names=None, edge_type='tx-tx'):
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
                        entry['target_cell'] = str(dst)
                
                data.append(entry)
    
    return pd.DataFrame(data)

def visualize_attention_df(attention_df, layer_idx, head_idx, edge_type, gene_names=None):
    """Visualize attention weights as a heatmap."""
    num_nodes = len(attention_df['source'].unique())
    filtered_df = attention_df[
        (attention_df['layer'] == layer_idx + 1) & 
        (attention_df['head'] == head_idx + 1)
    ]
    
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for _, row in filtered_df.iterrows():
        src = row['source']
        dst = row['target']
        adj_matrix[src, dst] = row['attention_weight']
    
    if gene_names is not None:
        sorted_indices = sorted(filtered_df['source'].unique(), key=lambda i: gene_names[i])
        adj_matrix = adj_matrix[sorted_indices][:, sorted_indices]
    
    figures_dir = create_figures_dir()
    save_path = figures_dir / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type}.png'
    
    plot_heatmap(
        matrix=adj_matrix,
        title=f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}',
        xlabel='Target Node (Transcript)',
        ylabel='Source Node (Transcript)',
        save_path=save_path
    )

def visualize_attention_difference(attention_df, edge_type, compare_type='layers', 
                                 layer_indices=None, head_indices=None, gene_names=None, 
                                 max_matrix_size=1000, downsample_threshold=500):
    """Visualize differences in attention weights across layers or heads."""
    filtered_df = attention_df[attention_df['edge_type'] == edge_type]
    
    if layer_indices is None:
        layer_indices = sorted(filtered_df['layer'].unique())
    if head_indices is None:
        head_indices = sorted(filtered_df['head'].unique())
    
    figures_dir = create_figures_dir()
    
    if compare_type == 'layers':
        _visualize_layer_differences(filtered_df, layer_indices, head_indices, gene_names, 
                                   max_matrix_size, downsample_threshold, figures_dir, edge_type)
    elif compare_type == 'heads':
        _visualize_head_differences(filtered_df, layer_indices, head_indices, gene_names, 
                                  max_matrix_size, downsample_threshold, figures_dir, edge_type)
    else:
        raise ValueError(f"Invalid comparison type: {compare_type}. Must be 'layers' or 'heads'.")

def _visualize_layer_differences(filtered_df, layer_indices, head_indices, gene_names, 
                               max_matrix_size, downsample_threshold, figures_dir, edge_type):
    """Helper function to visualize differences between layers."""
    for head_idx in head_indices:
        head_df = filtered_df[filtered_df['head'] == head_idx]
        if head_df.empty or len(layer_indices) < 2:
            continue
        
        num_nodes = len(head_df['source'].unique())
        if num_nodes == 0:
            continue
        
        if num_nodes > max_matrix_size:
            all_nodes = sorted(head_df['source'].unique())
            np.random.seed(42)
            sampled_nodes = np.random.choice(all_nodes, size=max_matrix_size, replace=False)
            head_df = head_df[head_df['source'].isin(sampled_nodes) & 
                            head_df['target'].isin(sampled_nodes)]
            num_nodes = len(sampled_nodes)
        
        for i in range(len(layer_indices) - 1):
            for j in range(i + 1, len(layer_indices)):
                layer1, layer2 = layer_indices[i], layer_indices[j]
                
                adj_matrix1 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                adj_matrix2 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                
                layer1_df = head_df[head_df['layer'] == layer1]
                layer2_df = head_df[head_df['layer'] == layer2]
                
                if layer1_df.empty or layer2_df.empty:
                    continue
                
                unique_nodes = sorted(set(head_df['source'].unique()) | set(head_df['target'].unique()))
                node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
                
                for _, row in layer1_df.iterrows():
                    src = node_to_idx[row['source']]
                    dst = node_to_idx[row['target']]
                    adj_matrix1[src, dst] += row['attention_weight']
                
                for _, row in layer2_df.iterrows():
                    src = node_to_idx[row['source']]
                    dst = node_to_idx[row['target']]
                    adj_matrix2[src, dst] += row['attention_weight']
                
                diff_matrix = adj_matrix2 - adj_matrix1
                
                if gene_names is not None:
                    node_genes = {node: gene_names[node] for node in unique_nodes if node < len(gene_names)}
                    if node_genes:
                        sorted_idx_map = sorted(range(len(unique_nodes)), 
                                            key=lambda i: node_genes.get(unique_nodes[i], ""))
                        diff_matrix = diff_matrix[sorted_idx_map][:, sorted_idx_map]
                
                if max(diff_matrix.shape) > downsample_threshold:
                    diff_matrix = downsample_matrix(diff_matrix, downsample_threshold)
                
                vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
                if vmax == 0:
                    vmax = 1
                
                save_path = figures_dir / f'attention_diff_layer{layer2}vs{layer1}_head{head_idx}_{edge_type}.png'
                plot_heatmap(
                    matrix=diff_matrix,
                    title=f'Attention Weight Difference - Layer {layer2} vs Layer {layer1}, Head {head_idx}, Edge Type: {edge_type}',
                    xlabel='Target Node (Transcript)',
                    ylabel='Source Node (Transcript)',
                    save_path=save_path,
                    cmap='coolwarm',
                    vmin=-vmax,
                    vmax=vmax
                )

def _visualize_head_differences(filtered_df, layer_indices, head_indices, gene_names, 
                              max_matrix_size, downsample_threshold, figures_dir, edge_type):
    """Helper function to visualize differences between heads."""
    for layer_idx in layer_indices:
        layer_df = filtered_df[filtered_df['layer'] == layer_idx]
        if layer_df.empty or len(head_indices) < 2:
            continue
        
        num_nodes = len(layer_df['source'].unique())
        if num_nodes == 0:
            continue
        
        if num_nodes > max_matrix_size:
            all_nodes = sorted(layer_df['source'].unique())
            np.random.seed(42)
            sampled_nodes = np.random.choice(all_nodes, size=max_matrix_size, replace=False)
            layer_df = layer_df[layer_df['source'].isin(sampled_nodes) & 
                              layer_df['target'].isin(sampled_nodes)]
            num_nodes = len(sampled_nodes)
        
        for i in range(len(head_indices) - 1):
            for j in range(i + 1, len(head_indices)):
                head1, head2 = head_indices[i], head_indices[j]
                
                adj_matrix1 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                adj_matrix2 = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                
                head1_df = layer_df[layer_df['head'] == head1]
                head2_df = layer_df[layer_df['head'] == head2]
                
                if head1_df.empty or head2_df.empty:
                    continue
                
                unique_nodes = sorted(set(layer_df['source'].unique()) | set(layer_df['target'].unique()))
                node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
                
                for _, row in head1_df.iterrows():
                    src = node_to_idx[row['source']]
                    dst = node_to_idx[row['target']]
                    adj_matrix1[src, dst] += row['attention_weight']
                
                for _, row in head2_df.iterrows():
                    src = node_to_idx[row['source']]
                    dst = node_to_idx[row['target']]
                    adj_matrix2[src, dst] += row['attention_weight']
                
                diff_matrix = adj_matrix2 - adj_matrix1
                
                if gene_names is not None:
                    node_genes = {node: gene_names[node] for node in unique_nodes if node < len(gene_names)}
                    if node_genes:
                        sorted_idx_map = sorted(range(len(unique_nodes)), 
                                            key=lambda i: node_genes.get(unique_nodes[i], ""))
                        diff_matrix = diff_matrix[sorted_idx_map][:, sorted_idx_map]
                
                if max(diff_matrix.shape) > downsample_threshold:
                    diff_matrix = downsample_matrix(diff_matrix, downsample_threshold)
                
                vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
                if vmax == 0:
                    vmax = 1
                
                save_path = figures_dir / f'attention_diff_head{head2}vs{head1}_layer{layer_idx}_{edge_type}.png'
                plot_heatmap(
                    matrix=diff_matrix,
                    title=f'Attention Weight Difference - Head {head2} vs Head {head1}, Layer {layer_idx}, Edge Type: {edge_type}',
                    xlabel='Target Node (Transcript)',
                    ylabel='Source Node (Transcript)',
                    save_path=save_path,
                    cmap='coolwarm',
                    vmin=-vmax,
                    vmax=vmax
                ) 