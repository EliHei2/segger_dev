import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np
from .gene_embedding import gene_embedding
from .utils import create_figures_dir
import scanpy as sc
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from sklearn.cluster import AgglomerativeClustering

def clustering_gene_cell_umap(attention_matrix, gene_names, cell_ids, 
                                          n_neighbors_list=[5, 15, 30], 
                                          n_components_list=[2, 3, 5], min_dist=0.1,
                                          n_gene_clusters=5, n_cell_clusters=3,
                                          metric='cosine', 
                                          random_state=42, visualization=True, method='hierarchical',
                                          path_suffix=''):
    """
    Perform clustering on UMAP embeddings for gene-cell attention matrices.
    
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
    method : str
        Method for clustering, either 'hierarchical' or 'leiden'
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
            # Perform clustering on cells (columns)
            cell_results = _cluster_cells(attention_matrix, cell_ids, n_neighbors, 
                                        n_components, min_dist, n_cell_clusters, 
                                        random_state, visualization, metric,
                                        figures_dir, 
                                        f'n_neighbors_{n_neighbors}_n_components_{n_components}_{path_suffix}',
                                        method)
            
            # Perform clustering on genes (rows)
            gene_results = _cluster_genes(attention_matrix, gene_names, n_neighbors, 
                                        n_components, min_dist, n_gene_clusters, 
                                        random_state, visualization, metric,
                                        figures_dir, 
                                        f'n_neighbors_{n_neighbors}_n_components_{n_components}_{path_suffix}',
                                        method)
            
            # Store results
            results[f'n_neighbors_{n_neighbors}_n_components_{n_components}'] = {
                'cell_clustering': cell_results,
                'gene_clustering': gene_results
            }
    
    return results

def _cluster_cells(attention_matrix, cell_ids, n_neighbors, n_components, min_dist, 
                  n_clusters, random_state, visualization, metric,
                  figures_dir, param_suffix, method='hierarchical', leiden_resolution=0.5, path_suffix=''):
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
    
    if method == 'leiden':
        sc.tl.leiden(adata_cells, resolution=leiden_resolution, random_state=random_state)
        # Convert string labels to sequential integers to avoid indexing issues
        leiden_labels = adata_cells.obs['leiden'].values
        unique_labels = np.unique(leiden_labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        cluster_labels = np.array([label_mapping[label] for label in leiden_labels])
        # You can skip hierarchical clustering if using Leiden
        linkage_matrix = None
        leaf_order = np.argsort(cluster_labels)
    elif method == 'hierarchical':
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
        leaf_order = leaves_list(linkage_matrix)
    else:
        raise ValueError(f"Invalid method: {method}")
    
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

def _cluster_genes(attention_matrix, gene_names, n_neighbors, n_components, min_dist, 
                  n_clusters, random_state, visualization, metric,
                  figures_dir, param_suffix, method='hierarchical', leiden_resolution=0.5, path_suffix=''):
    """Perform hierarchical clustering on genes based on their attention patterns from cells."""
    
    # Transpose matrix to cluster genes (now rows) based on cell attention patterns
    attention_matrix_T = attention_matrix
    
    # Create AnnData object for genes
    adata_genes = sc.AnnData(attention_matrix_T)
    
    if attention_matrix_T.shape[0] > 50:
        # Perform PCA preprocessing
        sc.pp.pca(adata_genes, n_comps=min(50, attention_matrix_T.shape[0]), random_state=random_state)
    
    # Perform UMAP
    sc.pp.neighbors(adata_genes, n_neighbors=min(n_neighbors, attention_matrix_T.shape[1]-1), 
                   random_state=random_state, use_rep='X_pca', metric=metric)
    sc.tl.umap(adata_genes, n_components=n_components, min_dist=min_dist, random_state=random_state)
    
    # Get UMAP coordinates
    coordinates = adata_genes.obsm['X_umap']
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(coordinates, method='average', metric=metric)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Get leaf ordering for dendrogram
    leaf_order = leaves_list(linkage_matrix)
    
    results = {
        'coordinates': coordinates,
        'cluster_labels': cluster_labels,
        'linkage_matrix': linkage_matrix,
        'leaf_order': leaf_order
    }
    
    return results
    

def _create_cell_clustering_visualization(attention_matrix_T, coordinates, cluster_labels, 
                                        linkage_matrix, cell_ids, n_components, 
                                        figures_dir, filename):
    """Create visualization for cell clustering.
    If linkage_matrix is None (e.g., when using Leiden), dendrogram is skipped.
    """
    import matplotlib.gridspec as gridspec
    # If linkage_matrix is provided, plot dendrogram, heatmap, and embedding
    if linkage_matrix is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # Plot dendrogram and get leaf ordering
        leaf_order = leaves_list(linkage_matrix)
    else:
        # Only plot heatmap and embedding
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        # Use cluster label ordering for leaf order
        leaf_order = np.argsort(cluster_labels)
    
    # Sort attention matrix by dendrogram leaf ordering
    sorted_matrix = attention_matrix_T[leaf_order]
    sorted_labels = cluster_labels[leaf_order]
    sorted_cell_ids = [cell_ids[i] for i in leaf_order]
    
    # Create heatmap
    sns.heatmap(sorted_matrix, ax=ax1, cmap='viridis', 
               xticklabels=False, yticklabels=False)
    ax1.set_title('Cell-Gene Attention Matrix (Cells Clustered)')
    
    # Add cluster boundaries (only if we have multiple clusters)
    unique_labels = np.unique(sorted_labels)
    if len(unique_labels) > 1:
        for label in unique_labels[:-1]:
            boundary = np.where(sorted_labels == label)[0][-1]+1
            ax1.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
    
    # Plot the cell embedding
    scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, 
                        cmap='tab10', alpha=0.6)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    ax2.set_title('Cell Embedding')
    
    # Add legend for cell clusters
    legend1 = ax2.legend(*scatter.legend_elements(),
                       title="Cell Clusters", loc="upper right")
    ax2.add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'clustering' / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_gene_cell_cluster_summary(attention_matrix,
                                   gene_cluster_labels, cell_cluster_labels, 
                                   figures_dir, filename='gene_cell_cluster_summary', reduced_gene=False, normalized_gene=False,
                                   gene_linkage_matrix=None, cell_linkage_matrix=None, metric='cosine', gene_names=None):
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
    gene_names : list, optional
        List of gene names
    """
    
    # Safety checks
    if attention_matrix.shape[0] != len(gene_cluster_labels):
        raise ValueError(f"Gene cluster labels length ({len(gene_cluster_labels)}) doesn't match attention matrix rows ({attention_matrix.shape[0]})")
    
    if attention_matrix.shape[1] != len(cell_cluster_labels):
        raise ValueError(f"Cell cluster labels length ({len(cell_cluster_labels)}) doesn't match attention matrix columns ({attention_matrix.shape[1]})")
    
    if len(gene_cluster_labels) == 0 or len(cell_cluster_labels) == 0:
        print("Warning: Empty cluster labels provided")
        return None, None

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

    # Gene+Cell clustering heatmap (genes and cells sorted by dendrogram ordering if available)
    if gene_linkage_matrix is not None:
        # Get gene ordering from dendrogram
        sort_gene_idx = leaves_list(gene_linkage_matrix)
    else:
        # Fall back to cluster label ordering
        sort_gene_idx = np.argsort(gene_cluster_labels)
    
    if cell_linkage_matrix is not None:
        # Get cell ordering from dendrogram
        sort_cell_idx = leaves_list(cell_linkage_matrix)
        # Safety check: ensure ordering is within bounds
        if len(sort_cell_idx) > len(cell_cluster_labels):
            sort_cell_idx = sort_cell_idx[:len(cell_cluster_labels)]
    else:
        # Fall back to cluster label ordering
        sort_cell_idx = np.argsort(cell_cluster_labels)
    
    # Sort the matrix by gene and cell cluster labels
    sorted_gene_cell_matrix = attention_matrix[sort_gene_idx][:, sort_cell_idx]
    sorted_gene_labels = gene_cluster_labels[sort_gene_idx]
    # count the number of each gene label appears in sorted_gene_labels
    gene_label_counts = np.zeros(len(unique_gene_labels))
    for i in range(len(unique_gene_labels)):
        gene_label_counts[i] = np.sum(sorted_gene_labels == unique_gene_labels[i])
    print('gene_label_counts', gene_label_counts)
    sorted_gene_names = np.array(gene_names)[sort_gene_idx]
    sorted_cell_labels = cell_cluster_labels[sort_cell_idx]
    
    if reduced_gene:
        # Optionally reduce to top 30 genes by attention sum
        gene_sums = sorted_gene_cell_matrix.sum(axis=1)
        top_gene_indices = np.argsort(gene_sums)[-30:][::-1]
        
        # Apply the reduction to both matrix and labels
        sorted_gene_cell_matrix = sorted_gene_cell_matrix[top_gene_indices, :]
        # Keep the order of sorted_gene_labels but extract the top 30 gene names
        sorted_top_gene_names = []
        for i in range(len(sort_gene_idx)):
            if i in top_gene_indices:
                sorted_top_gene_names.append(sorted_gene_names[i])
        
        # Update sort_gene_idx to reflect the reduction
        sort_gene_idx = sort_gene_idx[top_gene_indices]
        print('sorted_top_gene_names', sorted_top_gene_names)

    if normalized_gene:
        # normalize the attention matrix
        sorted_gene_cell_matrix = sorted_gene_cell_matrix / (sorted_gene_cell_matrix.sum(axis=0, keepdims=True) + 1e-10)
        filename = f'{filename}_normalized_gene'
    
    # Create figure
    plt.figure(figsize=(15, 8))

    # Determine yticklabels to use
    if reduced_gene:
        yticklabels = sorted_top_gene_names
    else:
        yticklabels = sorted_gene_names

    sns.heatmap(sorted_gene_cell_matrix, cmap='viridis', vmin=0.2, vmax=0.5, xticklabels=False, yticklabels=yticklabels)
    plt.title('Gene-Cell Normalized Attention Matrix (Genes & Cells Clustered)')
    plt.yticks(rotation=0, fontsize=10, ha='right')
    # Add cell cluster boundaries (vertical) - only if we have multiple clusters
    unique_cell_labels = np.unique(sorted_cell_labels)
    if len(unique_cell_labels) > 1:
        for label in unique_cell_labels[:-1]:
            boundary = np.where(sorted_cell_labels == label)[0][-1]+1
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
                
    # Add cell cluster labels to heatmap axes
    label_changes = np.where(np.diff(sorted_cell_labels) != 0)[0] + 1
    label_positions = np.concatenate([[0], label_changes, [len(sorted_cell_labels)]])
    
    # Add x-axis labels (cell clusters)
    for i, label in enumerate(unique_cell_labels):
        start_pos = label_positions[i]
        end_pos = label_positions[i + 1]
        mid_pos = (start_pos + end_pos) / 2
        plt.text(mid_pos, len(sorted_cell_labels) * 0.0033, f'Cell Cluster {label+1}', 
                rotation=45, ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'clustering' / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return gene_cluster_stats, cell_cluster_stats