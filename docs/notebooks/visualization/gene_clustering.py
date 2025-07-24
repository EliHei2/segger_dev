import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np
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
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
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
            
            # Get the leaf order from dendrogram (hierarchical ordering)
            leaf_order = leaves_list(linkage_matrix)
            
            # Store results
            results[f'n_neighbors_{n_neighbors}_n_components_{n_components}'] = {
                'coordinates': coordinates,
                'cluster_labels': cluster_labels,
                'linkage_matrix': linkage_matrix,
                'leaf_order': leaf_order
            }
            
            if visualization:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot heatmap of attention matrix with cluster labels
                if isinstance(attention_matrix, (lil_matrix)):
                    attention_matrix = attention_matrix.toarray()
                
                # Sort attention matrix by cluster labels
                sort_idx = np.argsort(cluster_labels)
                sorted_matrix = attention_matrix[sort_idx][:, sort_idx]
                sorted_labels = cluster_labels[sort_idx]
        
                # Create heatmap
                sns.heatmap(sorted_matrix, ax=ax1, cmap='viridis', 
                           xticklabels=False, yticklabels=False)
                ax1.set_title('Attention Matrix Heatmap (Clustered)')
                
                # Add cluster boundaries
                unique_labels = np.unique(sorted_labels)
                for label in unique_labels[:-1]:
                    boundary = np.where(sorted_labels == label)[0][-1]+1
                    ax1.axhline(y=boundary, color='red', linestyle='--', alpha=0.5)
                    ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
                
                # Add the gene cluster labels to the heatmap
                # Find positions where cluster labels change
                label_changes = np.where(np.diff(sorted_labels) != 0)[0] + 1
                label_positions = np.concatenate([[0], label_changes, [len(sorted_labels)]])
                
                # Add y-axis labels (gene clusters)
                for i, label in enumerate(unique_labels):
                    start_pos = label_positions[i]
                    end_pos = label_positions[i + 1]
                    mid_pos = (start_pos + end_pos) / 2
                    ax1.text(-len(sorted_labels) * 0.02, mid_pos, f'Cluster {label}', 
                            rotation=0, ha='right', va='center', fontsize=10, fontweight='bold')
                
                # Plot the embedding with gene cluster labels
                scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
                ax2.set_xlabel('UMAP 1')
                ax2.set_ylabel('UMAP 2')
                
                ax2.set_title('Gene Embedding')
                
                # Add legend for gene clusters
                legend1 = ax2.legend(*scatter.legend_elements(),
                                   title="Gene Clusters", loc="upper right")
                ax2.add_artist(legend1)
                
                plt.tight_layout()
                # plt.savefig(figures_dir / 'clustering' / f'hierarchical_clustering_umap_{n_neighbors}_{n_components}.png')
                # plt.close()
    
    return results 