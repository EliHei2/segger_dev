import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np
from .gene_embedding import gene_embedding
from .utils import create_figures_dir
import scanpy as sc
import seaborn as sns

def hierarchical_clustering_umap(attention_matrix, n_neighbors_list=[5, 15, 30], 
                               n_components_list=[2, 3, 5], min_dist=0.1,
                               n_clusters=5, random_state=42, visualization=True):
    """
    Perform hierarchical clustering on UMAP embeddings with different parameters.
    
    Parameters:
    -----------
    attention_matrix : scipy.sparse.lil_matrix or numpy.ndarray
        The attention matrix of shape (num_genes, num_genes)
    n_neighbors_list : list
        List of different n_neighbors values for UMAP
    n_components_list : list
        List of different n_components values for UMAP
    min_dist : float
        Minimum distance parameter for UMAP
    n_clusters : int
        Number of clusters for hierarchical clustering
    random_state : int
        Random seed for reproducibility
    visualization : bool
        Whether to create and return dendrogram visualization plots
        
    Returns:
    --------
    dict
        Dictionary containing embeddings and clustering results for each parameter combination
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from sklearn.cluster import AgglomerativeClustering
    
    # Convert sparse matrix to dense if necessary
    if isinstance(attention_matrix, (lil_matrix)):
        attention_matrix = attention_matrix.toarray()
    
    # Create AnnData object
    adata = sc.AnnData(attention_matrix)
    
    # Always perform PCA preprocessing first
    sc.pp.pca(adata, n_comps=50, random_state=random_state)
    
    results = {}
    
    # Create figures directory
    figures_dir = create_figures_dir(edge_type='tx-tx')
    
    for n_neighbors in n_neighbors_list:
        for n_components in n_components_list:
            # Perform UMAP
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state, use_rep='X_pca')
            sc.tl.umap(adata, n_components=n_components, min_dist=min_dist, random_state=random_state)
            
            # Get UMAP coordinates
            umap_key = f'X_umap_{n_neighbors}_{n_components}'
            coordinates = adata.obsm['X_umap']
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(coordinates, method='ward')
            # clustering = AgglomerativeClustering(n_clusters=n_clusters)
            # cluster_labels = clustering.fit_predict(coordinates)
            
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Store results
            results[f'n_neighbors_{n_neighbors}_n_components_{n_components}'] = {
                'coordinates': coordinates,
                'cluster_labels': cluster_labels,
                'linkage_matrix': linkage_matrix
            }
            
            if visualization:
                # Create figure with two subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
                
                # Plot dendrogram
                dendrogram(linkage_matrix, ax=ax1, truncate_mode='level', p=5)
                ax1.set_title('Hierarchical Clustering Dendrogram')

                # Plot heatmap of attention matrix with cluster labels
                if isinstance(attention_matrix, (lil_matrix)):
                    attention_matrix = attention_matrix.toarray()
                
                # Sort attention matrix by cluster labels
                sort_idx = np.argsort(cluster_labels)
                sorted_matrix = attention_matrix[sort_idx][:, sort_idx]
                sorted_labels = cluster_labels[sort_idx]
        
                # Create heatmap
                sns.heatmap(sorted_matrix, ax=ax2, cmap='viridis', 
                           xticklabels=False, yticklabels=False)
                ax2.set_title('Attention Matrix Heatmap (Clustered)')
                
                # Add cluster boundaries
                unique_labels = np.unique(sorted_labels)
                for label in unique_labels[:-1]:
                    boundary = np.where(sorted_labels == label)[0][-1]+1
                    ax2.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
                    ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
                
                # Plot the embedding
                if n_components == 2:
                    ax3.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
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
                plt.savefig(figures_dir / 'clustering' / f'hierarchical_clustering_umap_{n_neighbors}_{n_components}.png')
                plt.close()
    
    return results 