#!/usr/bin/env python3
"""
Test script to verify that dendrogram ordering is working correctly in the heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from docs.notebooks.visualization.gene_cell_clustering import (
    hierarchical_clustering_gene_cell_umap,
    create_gene_cell_cluster_summary
)

def create_test_data(n_genes=70, n_cells=100):
    """Create synthetic test data for clustering."""
    np.random.seed(42)
    
    # Create a synthetic attention matrix with some structure
    attention_matrix = np.random.rand(n_genes, n_cells)
    
    # Add some cluster structure
    # Create 3 gene clusters
    gene_cluster_size = n_genes // 3
    for i in range(3):
        start_idx = i * gene_cluster_size
        end_idx = min((i + 1) * gene_cluster_size, n_genes)
        # Make genes in same cluster more similar
        cluster_pattern = np.random.rand(n_cells)
        for j in range(start_idx, end_idx):
            attention_matrix[j, :] = cluster_pattern + np.random.normal(0, 0.1, n_cells)
    
    # Create 4 cell clusters
    cell_cluster_size = n_cells // 4
    for i in range(4):
        start_idx = i * cell_cluster_size
        end_idx = min((i + 1) * cell_cluster_size, n_cells)
        # Make cells in same cluster more similar
        cluster_pattern = np.random.rand(n_genes)
        for j in range(start_idx, end_idx):
            attention_matrix[:, j] = cluster_pattern + np.random.normal(0, 0.1, n_genes)
    
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    cell_ids = [f"Cell_{i}" for i in range(n_cells)]
    
    return attention_matrix, gene_names, cell_ids

def test_dendrogram_ordering():
    """Test that dendrogram ordering is working correctly."""
    
    print("Creating test data...")
    attention_matrix, gene_names, cell_ids = create_test_data(n_genes=50, n_cells=100)
    
    print("Running hierarchical clustering...")
    results = hierarchical_clustering_gene_cell_umap(
        attention_matrix=attention_matrix,
        gene_names=gene_names,
        cell_ids=cell_ids,
        n_neighbors_list=[15],
        n_components_list=[2],
        n_gene_clusters=3,
        n_cell_clusters=4,
        random_state=42,
        visualization=True
    )
    
    # Get results from the first parameter combination
    first_key = list(results.keys())[0]
    gene_results = results[first_key]['gene_clustering']
    cell_results = results[first_key]['cell_clustering']
    
    # Create figures directory
    figures_dir = Path('test_figures')
    figures_dir.mkdir(exist_ok=True)
    (figures_dir / 'clustering').mkdir(exist_ok=True)
    
    print("Creating summary with dendrogram ordering...")
    gene_stats, cell_stats = create_gene_cell_cluster_summary(
        attention_matrix=attention_matrix,
        gene_cluster_labels=gene_results['cluster_labels'],
        cell_cluster_labels=cell_results['cluster_labels'],
        figures_dir=figures_dir,
        filename='test_dendrogram_ordering',
        reduced_gene=False,
        gene_linkage_matrix=gene_results['linkage_matrix'],
        cell_linkage_matrix=cell_results['linkage_matrix']
    )
    
    print("Creating summary without dendrogram ordering (for comparison)...")
    gene_stats_old, cell_stats_old = create_gene_cell_cluster_summary(
        attention_matrix=attention_matrix,
        gene_cluster_labels=gene_results['cluster_labels'],
        cell_cluster_labels=cell_results['cluster_labels'],
        figures_dir=figures_dir,
        filename='test_cluster_label_ordering',
        reduced_gene=False,
        gene_linkage_matrix=None,
        cell_linkage_matrix=None
    )
    
    print(f"\nTest completed!")
    print(f"Figures saved to: {figures_dir}")
    print(f"Check the following files to compare ordering:")
    print(f"  - {figures_dir}/clustering/test_dendrogram_ordering.png (with dendrogram ordering)")
    print(f"  - {figures_dir}/clustering/test_cluster_label_ordering.png (with cluster label ordering)")
    
    # Print some statistics
    print(f"\nGene cluster sizes: {gene_stats['Size'].tolist()}")
    print(f"Cell cluster sizes: {cell_stats['Size'].tolist()}")
    
    return results

if __name__ == "__main__":
    test_dendrogram_ordering() 