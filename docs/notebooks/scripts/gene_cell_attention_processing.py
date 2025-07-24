#!/usr/bin/env python3
"""
Script to run gene-cell clustering on processed attention matrices.
This script demonstrates how to use the gene_cell_clustering module with real data.
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from docs.notebooks.visualization.gene_cell_clustering import (
    clustering_gene_cell_umap
)

def load_attention_data(edge_type='tx-bd'):
    """Load the processed attention data."""
    results_dir = Path('results')
    attention_file = results_dir / f'attention_gene_matrix_dict_{edge_type}_10000.pkl'
    
    if not attention_file.exists():
        raise FileNotFoundError(f"Attention data file not found: {attention_file}")
    
    with open(attention_file, 'rb') as f:
        attention_data = pickle.load(f)
    
    return attention_data

def run_gene_cell_clustering(attention_data, max_genes=500, max_cells=10000, normalized=True, metric='cosine', layer_index=0, method='leiden'):
    """Run gene-cell clustering on the attention data."""
    
    # Get the overall average attention matrix across all layers and heads
    layers = len(attention_data['adj_matrix'])
    heads = len(attention_data['adj_matrix'][0])
    
    # Compute average attention matrix
    avg_matrix = np.zeros_like(attention_data['adj_matrix'][0][0].toarray())
    if layer_index is not None:
        for head_idx in range(heads):
            avg_matrix += attention_data['adj_matrix'][layer_index][head_idx].toarray()
    else:
        for layer_idx in range(layers):
            for head_idx in range(heads):
                avg_matrix += attention_data['adj_matrix'][layer_idx][head_idx].toarray()
    avg_matrix = avg_matrix / (layers * heads)
    
    # Get gene names and cell IDs
    gene_names = attention_data['gene_names']
    cell_ids = attention_data['cell_ids']
    
    # Subsample if needed
    if len(cell_ids) > max_cells:
        np.random.seed(42)
        cell_indices = np.random.choice(len(cell_ids), max_cells, replace=False)
        avg_matrix = avg_matrix[:, cell_indices]
        cell_ids = [cell_ids[i] for i in cell_indices]
    
    # Optionally reduce to top max_genes genes by attention sum
    if max_genes:
        gene_sums = avg_matrix.sum(axis=1)
        top_gene_indices = np.argsort(gene_sums)[-max_genes:][::-1]
        avg_matrix = avg_matrix[top_gene_indices, :]
        gene_names = [gene_names[i] for i in top_gene_indices]
    
    if normalized:
        avg_matrix = avg_matrix / (avg_matrix.sum(axis=0, keepdims=True) + 1e-10)
    
    # Run hierarchical clustering (no visualization)
    results = clustering_gene_cell_umap(
        attention_matrix=avg_matrix,
        gene_names=gene_names,
        cell_ids=cell_ids,
        n_neighbors_list=[15],
        n_components_list=[3],
        n_gene_clusters=14,
        n_cell_clusters=9,
        metric=metric,
        random_state=42,
        visualization=False,
        method=method,
        path_suffix=f'layer_{layer_index+1}'
    )
    
    return results, avg_matrix, gene_names, cell_ids

def main():
    """Main function to run gene-cell clustering analysis."""
    
    print("Loading attention data...")
    try:
        attention_data = load_attention_data(edge_type='tx-bd')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_dataset.py first to generate the attention data.")
        return
    
    layer_indices = [4]
    # If not exist, run the gene-cell clustering
    clustering_result_path = Path('intermediate_data') / f'clustering_results_tx-bd_layer_{layer_indices[0]}.pkl'
    if not clustering_result_path.exists():
        print("Running gene-cell clustering...")
        for layer_index in layer_indices:
            results, attention_matrix, gene_names, cell_ids = run_gene_cell_clustering(
                attention_data, 
                normalized=True,
                max_genes=None,
                max_cells=10000,
                layer_index=layer_index,
                method='leiden'
            )
            # Save results with cell ids
            output_path = Path('intermediate_data') / f'clustering_results_tx-bd_layer_{layer_index}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump({'results': results, 'attention_matrix': attention_matrix, 'gene_names': gene_names, 'cell_ids': cell_ids}, f)
            print(f"Gene-cell clustering results saved to {output_path}")
    else:
        print(f"Gene-cell clustering results already exist at {clustering_result_path}")

if __name__ == "__main__":
    main() 