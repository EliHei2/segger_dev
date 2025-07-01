import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np
from .gene_embedding import gene_embedding
from .utils import create_figures_dir
import scanpy as sc
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering_gene_cell_umap(attention_matrix, gene_names, cell_ids, 
                                          n_neighbors_list=[5, 15, 30], 
                                          n_components_list=[2, 3, 5], min_dist=0.1,
                                          n_gene_clusters=5, n_cell_clusters=3,
                                          metric='cosine',
                                          random_state=42, visualization=True):
    """
    Perform hierarchical clustering on UMAP embeddings for gene-cell attention matrices.
    
    Parameters:
    -----------
    attention_matrix : scipy.sparse.lil_matrix or numpy.ndarray
        The attention matrix of shape (num_genes, num_cells)
    gene_names : List[str]
        List of gene names corresponding to rows
    cell_ids : List[str]
        List of cell IDs corresponding to columns
    n_neighbors_list : list
        List of different n_neighbors values for UMAP
    n_components_list : list
        List of different n_components values for UMAP
    min_dist : float
        Minimum distance parameter for UMAP
    n_gene_clusters : int
        Number of gene clusters for hierarchical clustering
    n_cell_clusters : int
        Number of cell clusters for hierarchical clustering
    metric : str
        Metric for hierarchical clustering, e.g. 'cosine', 'euclidean', 'manhattan'
    random_state : int
        Random seed for reproducibility
    visualization : bool
        Whether to create and return dendrogram visualization plots
        
    Returns:
    --------
    dict
        Dictionary containing embeddings and clustering results for each parameter combination
    """
    
    # Convert sparse matrix to dense if necessary
    if isinstance(attention_matrix, (lil_matrix)):
        attention_matrix = attention_matrix.toarray()
    
    # Create figures directory
    figures_dir = create_figures_dir(edge_type='tx-bd')
    
    results = {}
    
    for n_neighbors in n_neighbors_list:
        for n_components in n_components_list:
            # Perform clustering on genes (rows)
            gene_results = _cluster_genes(attention_matrix, gene_names, n_neighbors, 
                                        n_components, min_dist, n_gene_clusters, 
                                        random_state, visualization, metric,
                                        figures_dir, 
                                        f'n_neighbors_{n_neighbors}_n_components_{n_components}')
            
            # Perform clustering on cells (columns)
            cell_results = _cluster_cells(attention_matrix, cell_ids, n_neighbors, 
                                        n_components, min_dist, n_cell_clusters, 
                                        random_state, visualization, metric,
                                        figures_dir, 
                                        f'n_neighbors_{n_neighbors}_n_components_{n_components}')
            
            # Store results
            results[f'n_neighbors_{n_neighbors}_n_components_{n_components}'] = {
                'gene_clustering': gene_results,
                'cell_clustering': cell_results
            }
    
    return results

def _cluster_genes(attention_matrix, gene_names, n_neighbors, n_components, min_dist, 
                  n_clusters, random_state, visualization, metric,
                  figures_dir, param_suffix):
    """Perform clustering on genes based on their attention patterns to cells."""
    
    # Create AnnData object for genes
    adata_genes = sc.AnnData(attention_matrix)
    
    if attention_matrix.shape[1] > 50:
        # Perform PCA preprocessing
        sc.pp.pca(adata_genes, n_comps=min(50, attention_matrix.shape[1]), random_state=random_state)
    
    # Perform UMAP
    sc.pp.neighbors(adata_genes, n_neighbors=min(n_neighbors, attention_matrix.shape[0]-1), 
                   random_state=random_state, use_rep='X_pca', metric=metric)
    sc.tl.umap(adata_genes, n_components=n_components, min_dist=min_dist, random_state=random_state)
    
    # Get UMAP coordinates
    coordinates = adata_genes.obsm['X_umap']
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(coordinates, method='average', metric=metric)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Get leaf ordering for dendrogram
    dendro_data = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro_data['leaves']
    
    results = {
        'coordinates': coordinates,
        'cluster_labels': cluster_labels,
        'linkage_matrix': linkage_matrix,
        'leaf_order': leaf_order
    }
    
    if visualization:
        _create_gene_clustering_visualization(attention_matrix, coordinates, cluster_labels, 
                                            linkage_matrix, gene_names, n_components, 
                                            figures_dir, f'gene_clustering_{param_suffix}_{metric}')
    
    return results

def _cluster_cells(attention_matrix, cell_ids, n_neighbors, n_components, min_dist, 
                  n_clusters, random_state, visualization, metric,
                  figures_dir, param_suffix):
    """Perform clustering on cells based on their attention patterns from genes."""
    
    # Transpose matrix to cluster cells (now rows) based on gene attention patterns
    attention_matrix_T = attention_matrix.T
    
    # Create AnnData object for cells
    adata_cells = sc.AnnData(attention_matrix_T)
    
    if attention_matrix_T.shape[1] > 50:
        # Perform PCA preprocessing
        sc.pp.pca(adata_cells, n_comps=min(50, attention_matrix_T.shape[1]), random_state=random_state)
    
    # Perform UMAP
    sc.pp.neighbors(adata_cells, n_neighbors=min(n_neighbors, attention_matrix_T.shape[0]-1), 
                   random_state=random_state, use_rep='X_pca', metric=metric)
    sc.tl.umap(adata_cells, n_components=n_components, min_dist=min_dist, random_state=random_state)
    
    # Get UMAP coordinates
    coordinates = adata_cells.obsm['X_umap']
    
    # --- Elbow plot for optimal cluster number (WCSS) ---
    if visualization:
        from scipy.spatial.distance import cdist
        wcss = []
        cluster_range = range(2, 16)
        linkage_matrix = linkage(coordinates, method='average', metric=metric)
        for k in cluster_range:
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            wcss_k = 0
            for cluster_id in np.unique(labels):
                cluster_points = coordinates[labels == cluster_id]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    wcss_k += np.sum((cluster_points - centroid) ** 2)
            wcss.append(wcss_k)
        plt.figure(figsize=(7, 5))
        plt.plot(list(cluster_range), wcss, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Plot for Cell Clustering')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / 'clustering' / f'elbow_plot_cells_{param_suffix}.png', dpi=300)
        plt.close()
    # --- End elbow plot ---
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(coordinates, method='average', metric=metric)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Get leaf ordering for dendrogram
    dendro_data = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro_data['leaves']
    
    results = {
        'coordinates': coordinates,
        'cluster_labels': cluster_labels,
        'linkage_matrix': linkage_matrix,
        'leaf_order': leaf_order
    }
    
    if visualization:
        _create_cell_clustering_visualization(attention_matrix_T, coordinates, cluster_labels, 
                                            linkage_matrix, cell_ids, n_components, 
                                            figures_dir, f'cell_clustering_{param_suffix}_{metric}')
    
    return results

def _create_gene_clustering_visualization(attention_matrix, coordinates, cluster_labels, 
                                        linkage_matrix, gene_names, n_components, 
                                        figures_dir, filename):
    """Create visualization for gene clustering."""
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot dendrogram and get leaf ordering
    dendro_data = dendrogram(linkage_matrix, ax=ax1, truncate_mode='level', p=5, no_plot=False)
    ax1.set_title('Gene Clustering Dendrogram')
    
    # Get the leaf ordering from dendrogram
    leaf_order = dendro_data['leaves']
    
    # Ensure leaf_order indices are within bounds
    max_idx = attention_matrix.shape[0] - 1
    if max(leaf_order) > max_idx:
        print(f"Warning: Leaf order contains index {max(leaf_order)} but matrix has {attention_matrix.shape[0]} rows")
        print(f"Leaf order range: {min(leaf_order)} to {max(leaf_order)}")
        # This shouldn't happen if the dendrogram is computed on the same data
        # Let's use the cluster label ordering as fallback
        leaf_order = np.argsort(cluster_labels)
    
    # Sort attention matrix by dendrogram leaf ordering
    sorted_matrix = attention_matrix[leaf_order]
    sorted_labels = cluster_labels[leaf_order]
    sorted_gene_names = [gene_names[i] for i in leaf_order]
    
    # Create heatmap
    sns.heatmap(sorted_matrix, ax=ax2, cmap='viridis', 
               xticklabels=False, yticklabels=False)
    ax2.set_title('Gene-Cell Attention Matrix (Genes Clustered)')
    
    # Add cluster boundaries
    unique_labels = np.unique(sorted_labels)
    for label in unique_labels[:-1]:
        boundary = np.where(sorted_labels == label)[0][-1]+1
        ax2.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
    
    # Plot the gene embedding
    if n_components == 2:
        scatter = ax3.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, 
                            cmap='tab10', alpha=0.6)
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
    elif n_components >= 3:
        # Create a new figure with 3D projection for the third subplot
        fig.delaxes(ax3)  # Remove the 2D axes
        ax3 = fig.add_subplot(133, projection='3d')  # Create new 3D axes
        scatter = ax3.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                            c=cluster_labels, cmap='tab10', alpha=0.6)
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        ax3.set_zlabel('UMAP 3')
    
    ax3.set_title('Gene Embedding')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'clustering' / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def _create_cell_clustering_visualization(attention_matrix_T, coordinates, cluster_labels, 
                                        linkage_matrix, cell_ids, n_components, 
                                        figures_dir, filename):
    """Create visualization for cell clustering."""
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot dendrogram and get leaf ordering
    dendro_data = dendrogram(linkage_matrix, ax=ax1, truncate_mode='level', p=5, no_plot=False)
    ax1.set_title('Cell Clustering Dendrogram')
    
    # Get the leaf ordering from dendrogram
    leaf_order = dendro_data['leaves']
    
    # Ensure leaf_order indices are within bounds
    max_idx = attention_matrix_T.shape[0] - 1
    if max(leaf_order) > max_idx:
        print(f"Warning: Leaf order contains index {max(leaf_order)} but matrix has {attention_matrix_T.shape[0]} rows")
        print(f"Leaf order range: {min(leaf_order)} to {max(leaf_order)}")
        # This shouldn't happen if the dendrogram is computed on the same data
        # Let's use the cluster label ordering as fallback
        leaf_order = np.argsort(cluster_labels)
    
    # Sort attention matrix by dendrogram leaf ordering
    sorted_matrix = attention_matrix_T[leaf_order]
    sorted_labels = cluster_labels[leaf_order]
    sorted_cell_ids = [cell_ids[i] for i in leaf_order]
    
    # Create heatmap
    sns.heatmap(sorted_matrix, ax=ax2, cmap='viridis', 
               xticklabels=False, yticklabels=False)
    ax2.set_title('Cell-Gene Attention Matrix (Cells Clustered)')
    
    # Add cluster boundaries
    unique_labels = np.unique(sorted_labels)
    for label in unique_labels[:-1]:
        boundary = np.where(sorted_labels == label)[0][-1]+1
        ax2.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
    
    # Plot the cell embedding
    if n_components == 2:
        scatter = ax3.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, 
                            cmap='tab10', alpha=0.6)
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
    elif n_components >= 3:
        # Create a new figure with 3D projection for the third subplot
        fig.delaxes(ax3)  # Remove the 2D axes
        ax3 = fig.add_subplot(133, projection='3d')  # Create new 3D axes
        scatter = ax3.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                            c=cluster_labels, cmap='tab10', alpha=0.6)
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        ax3.set_zlabel('UMAP 3')
    
    ax3.set_title('Cell Embedding')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'clustering' / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_gene_cell_cluster_summary(attention_matrix,
                                   gene_cluster_labels, cell_cluster_labels, 
                                   figures_dir, filename='gene_cell_cluster_summary', reduced_gene=False, normalized_gene=False,
                                   gene_linkage_matrix=None, cell_linkage_matrix=None, metric='cosine'):
    """
    Create a comprehensive summary visualization of gene and cell clusters.
    
    Parameters:
    -----------
    attention_matrix : numpy.ndarray
        The attention matrix of shape (num_genes, num_cells)
    gene_cluster_labels : numpy.ndarray
        Cluster labels for genes
    cell_cluster_labels : numpy.ndarray
        Cluster labels for cells
    figures_dir : Path
        Directory to save figures
    filename : str
        Filename for the saved figure
    reduced_gene : bool
        Whether to reduce to top 50 genes
    gene_linkage_matrix : numpy.ndarray, optional
        Linkage matrix for gene clustering (for dendrogram ordering)
    cell_linkage_matrix : numpy.ndarray, optional
        Linkage matrix for cell clustering (for dendrogram ordering)
    """

    # Create a table of gene cluster statistics
    unique_gene_labels, gene_counts = np.unique(gene_cluster_labels, return_counts=True)
    gene_cluster_stats = pd.DataFrame({
        'Cluster': unique_gene_labels,
        'Size': gene_counts
    })
    
    # Create a table of cell cluster statistics
    unique_cell_labels, cell_counts = np.unique(cell_cluster_labels, return_counts=True)
    cell_cluster_stats = pd.DataFrame({
        'Cluster': unique_cell_labels,
        'Size': cell_counts
    })
    
    if reduced_gene:
        # Optionally reduce to top 50 genes by attention sum
        gene_sums = attention_matrix.sum(axis=1)
        top_gene_indices = np.argsort(gene_sums)[-50:][::-1]
        attention_matrix = attention_matrix[top_gene_indices, :]
        gene_cluster_labels = gene_cluster_labels[top_gene_indices]
    if normalized_gene:
        # normalize the attention matrix
        attention_matrix = attention_matrix / (attention_matrix.sum(axis=0, keepdims=True) + 1e-10)
        filename = f'{filename}_normalized_gene'

    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Gene+Cell clustering heatmap (genes and cells sorted by dendrogram ordering if available)
    if gene_linkage_matrix is not None:
        # Get gene ordering from dendrogram
        gene_dendro_data = dendrogram(gene_linkage_matrix, no_plot=True)
        sort_gene_idx = gene_dendro_data['leaves']
        # Ensure indices are within bounds
        if max(sort_gene_idx) >= attention_matrix.shape[0]:
            print(f"Warning: Gene dendrogram indices out of bounds, using cluster label ordering")
            sort_gene_idx = np.argsort(gene_cluster_labels)
    else:
        # Fall back to cluster label ordering
        sort_gene_idx = np.argsort(gene_cluster_labels)
    
    if cell_linkage_matrix is not None:
        # Get cell ordering from dendrogram
        cell_dendro_data = dendrogram(cell_linkage_matrix, no_plot=True)
        sort_cell_idx = cell_dendro_data['leaves']
        # Ensure indices are within bounds
        if max(sort_cell_idx) >= attention_matrix.shape[1]:
            print(f"Warning: Cell dendrogram indices out of bounds, using cluster label ordering")
            sort_cell_idx = np.argsort(cell_cluster_labels)
    else:
        # Fall back to cluster label ordering
        sort_cell_idx = np.argsort(cell_cluster_labels)
    
    sorted_gene_cell_matrix = attention_matrix[sort_gene_idx][:, sort_cell_idx]
    sorted_gene_labels = gene_cluster_labels[sort_gene_idx]
    sorted_cell_labels = cell_cluster_labels[sort_cell_idx]
    sns.heatmap(sorted_gene_cell_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Gene-Cell Attention Matrix (Genes & Cells Clustered)')
    plt.ylabel('Genes')
    plt.xlabel('Cells')
    # Add gene cluster boundaries (horizontal)
    unique_gene_labels = np.unique(sorted_gene_labels)
    for label in unique_gene_labels[:-1]:
        boundary = np.where(sorted_gene_labels == label)[0][-1]+1
        plt.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
    # Add cell cluster boundaries (vertical)
    unique_cell_labels = np.unique(sorted_cell_labels)
    for label in unique_cell_labels[:-1]:
        boundary = np.where(sorted_cell_labels == label)[0][-1]+1
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(figures_dir / 'clustering' / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return gene_cluster_stats, cell_cluster_stats