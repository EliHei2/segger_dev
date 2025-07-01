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
    hierarchical_clustering_gene_cell_umap, 
    create_gene_cell_cluster_summary
)

def load_attention_data(edge_type='tx-bd'):
    """Load the processed attention data."""
    results_dir = Path('results')
    attention_file = results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl'
    
    if not attention_file.exists():
        raise FileNotFoundError(f"Attention data file not found: {attention_file}")
    
    with open(attention_file, 'rb') as f:
        attention_data = pickle.load(f)
    
    return attention_data

def run_gene_cell_clustering(attention_data, max_genes=500, max_cells=10000, metric='cosine'):
    """Run gene-cell clustering on the attention data."""
    
    # Get the overall average attention matrix across all layers and heads
    layers = len(attention_data['adj_matrix'])
    heads = len(attention_data['adj_matrix'][0])
    
    print(f"Processing attention data with {layers} layers and {heads} heads")
    
    # Compute average attention matrix
    avg_matrix = np.zeros_like(attention_data['adj_matrix'][0][0].toarray())
    for layer_idx in range(layers):
        for head_idx in range(heads):
            avg_matrix += attention_data['adj_matrix'][layer_idx][head_idx].toarray()
    
    avg_matrix = avg_matrix / (layers * heads)
    
    # Get gene names and cell IDs
    gene_names = attention_data['gene_names']
    cell_ids = attention_data['cell_ids']
    
    print(f"Original matrix shape: {avg_matrix.shape}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Number of cells: {len(cell_ids)}")
    
    # Subsample if needed
    if len(cell_ids) > max_cells:
        np.random.seed(42)
        cell_indices = np.random.choice(len(cell_ids), max_cells, replace=False)
        avg_matrix = avg_matrix[:, cell_indices]
        cell_ids = [cell_ids[i] for i in cell_indices]
        print(f"Subsampled to {len(cell_ids)} cells")
    
    # Optionally reduce to top max_genes genes by attention sum
    if max_genes:
        gene_sums = avg_matrix.sum(axis=1)
        top_gene_indices = np.argsort(gene_sums)[-max_genes:][::-1]
        avg_matrix = avg_matrix[top_gene_indices, :]
        gene_names = [gene_names[i] for i in top_gene_indices]
    
    print(f"Final matrix shape: {avg_matrix.shape}")
    
    # Run hierarchical clustering
    print("\nRunning gene-cell hierarchical clustering...")
    results = hierarchical_clustering_gene_cell_umap(
        attention_matrix=avg_matrix,
        gene_names=gene_names,
        cell_ids=cell_ids,
        n_neighbors_list=[30],
        n_components_list=[3],
        n_gene_clusters=5,
        n_cell_clusters=9,
        metric=metric,
        random_state=42,
        visualization=True
    )
    
    return results, avg_matrix, gene_names, cell_ids

def create_summary_visualizations(results, attention_matrix, metric):
    """Create summary visualizations for the clustering results."""
    
    # Get results from the first parameter combination
    first_key = list(results.keys())[0]
    gene_results = results[first_key]['gene_clustering']
    cell_results = results[first_key]['cell_clustering']
    
    # Create figures directory
    figures_dir = Path('figures') / 'tx-bd'
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / 'clustering').mkdir(exist_ok=True)
    
    print("\nCreating cluster summary...")
    gene_stats, cell_stats = create_gene_cell_cluster_summary(
        attention_matrix=attention_matrix,
        gene_cluster_labels=gene_results['cluster_labels'],
        cell_cluster_labels=cell_results['cluster_labels'],
        figures_dir=figures_dir,
        filename='gene_cell_cluster_summary',
        reduced_gene=True,
        normalized_gene=True,
        gene_linkage_matrix=gene_results['linkage_matrix'],
        cell_linkage_matrix=cell_results['linkage_matrix'],
        metric=metric
    )
    
    # # Print summary statistics
    # print("\nGene cluster statistics:")
    # for stat in gene_stats:
    #     print(f"  {stat['Cluster']}: {stat['Size']} genes")
    #     print(f"    Sample genes: {stat['Genes'][:5]}")
    
    # print("\nCell cluster statistics:")
    # for stat in cell_stats:
    #     print(f"  {stat['Cluster']}: {stat['Size']} cells")
    #     print(f"    Sample cells: {stat['Cells'][:5]}")
    
    print(f"\nFigures saved to: {figures_dir}")
    
    return gene_stats, cell_stats

def main():
    """Main function to run gene-cell clustering analysis."""
    
    print("Loading attention data...")
    try:
        attention_data = load_attention_data(edge_type='tx-bd')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_dataset.py first to generate the attention data.")
        return
    
    print("Running gene-cell clustering...")
    results, attention_matrix, gene_names, cell_ids = run_gene_cell_clustering(
        attention_data, 
        max_genes=500,  # Limit to the top 100 genes for better visualization
        max_cells=10000   # Limit to 5000 cells for faster processing
    )
    
    print("Creating summary visualizations...")
    gene_stats, cell_stats = create_summary_visualizations(
        results, attention_matrix, metric='cosine'
    )
    
    # Save results with cell ids
    with open(Path('intermediate_data') / 'clustering_results_tx-bd.pkl', 'wb') as f:
        pickle.dump({'results': results, 'attention_matrix': attention_matrix, 'gene_names': gene_names, 'cell_ids': cell_ids}, f)
    
    print("\nGene-cell clustering analysis completed successfully!")
    print(f"Results saved in: {Path('intermediate_data') / 'clustering_results_tx-bd.pkl'}")

if __name__ == "__main__":
    main() 