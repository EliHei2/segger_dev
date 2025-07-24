import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from docs.notebooks.visualization.gene_clustering import (
    hierarchical_clustering_umap
)

def main():
    # load attention gene matrix dict
    with open(Path('results') / f'attention_gene_matrix_dict_tx-tx.pkl', 'rb') as f:
        attention_gene_matrix_dict = pickle.load(f)
    
    # Get classified genes (it contains the gene names and groups)
    gene_types = pd.read_csv(Path('data_xenium') / 'gene_groups_ordered_color.csv')
    gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
    
    # Load transcripts
    transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')

    # Get gene names
    gene_names = transcripts['feature_name'].unique()
    
    # Add negative control genes to the gene_types_dict
    negative_control_genes = [
        'NegControlProbe_00034', 'NegControlProbe_00004', 'NegControlCodeword_0526',
        'NegControlCodeword_0517', 'NegControlProbe_00035', 'NegControlCodeword_0514',
        'NegControlCodeword_0506', 'NegControlCodeword_0534', 'NegControlProbe_00041',
        'NegControlCodeword_0513', 'NegControlCodeword_0531', 'NegControlCodeword_0522',
        'NegControlCodeword_0523', 'NegControlProbe_00022', 'NegControlProbe_00031',
        'NegControlCodeword_0530', 'NegControlCodeword_0508', 'NegControlCodeword_0511',
        'NegControlCodeword_0510', 'NegControlProbe_00024', 'NegControlProbe_00039',
        'NegControlProbe_00002', 'NegControlCodeword_0528', 'NegControlCodeword_0540',
        'NegControlCodeword_0503', 'NegControlCodeword_0536', 'NegControlCodeword_0539',
        'NegControlProbe_00013', 'NegControlCodeword_0520', 'NegControlCodeword_0524',
        'NegControlCodeword_0533', 'NegControlProbe_00042', 'BLANK_0069',
        'NegControlProbe_00025', 'NegControlProbe_00017', 'NegControlCodeword_0502',
        'NegControlProbe_00003', 'NegControlCodeword_0515', 'NegControlCodeword_0537',
        'NegControlProbe_00012', 'NegControlProbe_00016', 'NegControlCodeword_0521',
        'NegControlCodeword_0507', 'NegControlCodeword_0529', 'NegControlProbe_00033',
        'NegControlCodeword_0505', 'NegControlCodeword_0519', 'NegControlCodeword_0509',
        'NegControlCodeword_0500', 'NegControlCodeword_0538', 'NegControlProbe_00014',
        'NegControlCodeword_0516', 'NegControlCodeword_0535', 'NegControlCodeword_0527',
        'NegControlCodeword_0504', 'NegControlCodeword_0525', 'NegControlCodeword_0512',
        'BLANK_0037', 'NegControlCodeword_0518', 'NegControlCodeword_0532',
        'NegControlProbe_00019', 'BLANK_0006', 'NegControlCodeword_0501'
    ]
    gene_types_dict.update(dict(zip(negative_control_genes, ['negative_control'] * len(negative_control_genes))))

    # Sum across layers and heads to derive the attention matrix
    attention_matrix = np.zeros_like(attention_gene_matrix_dict['adj_matrix'][0][0].toarray())
    n_layers = 5
    n_heads = 4
    for layer_idx in [4]:
        for head_idx in range(n_heads):
            attention_matrix += attention_gene_matrix_dict['adj_matrix'][layer_idx][head_idx].toarray()

    # Hierarchical clustering (no visualization)
    clustering_results = hierarchical_clustering_umap(
        attention_matrix,
        n_neighbors_list=[15],
        n_components_list=[5],
        min_dist=0.1,
        n_clusters=5,
        random_state=42,
        visualization=False
    )

    # Save clustering results
    output_path = Path('intermediate_data') / f'clustering_results_tx-tx.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(clustering_results, f)
    print(f"Clustering results saved to {output_path}")

if __name__ == "__main__":
    main()