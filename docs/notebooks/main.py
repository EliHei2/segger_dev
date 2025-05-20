import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


import torch
import pandas as pd
import pickle
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import load_model
from docs.notebooks.visualization.transcript_visualization import extract_attention_df, visualize_attention_difference
from docs.notebooks.visualization.gene_visualization import summarize_attention_by_gene_df, compare_attention_patterns, visualize_all_attention_patterns, visualize_all_attention_patterns_with_metrics
from docs.notebooks.visualization.gene_embedding import visualize_attention_embedding, visualize_all_embeddings, visualize_average_embedding
from docs.notebooks.visualization.utils import safe_divide_sparse_numpy,get_top_genes_across_all_layers
from scipy.sparse import lil_matrix
import numpy as np

def main():
    # Paths to data and models
    model_version = 1
    model_path = Path('models') / "lightning_logs" / f"version_{model_version}"
    ls = load_model(model_path / "checkpoints")
    ls.eval()

    # Load transcripts
    transcripts = pd.read_parquet(Path('data_xenium') / 'transcripts.parquet')

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ls = ls.to(device)

    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=Path('data_segger'),
        batch_size=2,
        num_workers=2,
    )
    dm.setup()

    # Get a sample batch
    batch = dm.train[0].to(device)
    
    # Get gene names
    transcript_ids = batch['tx'].id.cpu().numpy()
    id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    gene_names_batch = [id_to_gene[id] for id in transcript_ids] # list of gene names for the batch one

    # Run forward pass to get attention weights
    with torch.no_grad():
        hetero_model = ls.model
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        _, attention_weights = hetero_model(x_dict, edge_index_dict)

    edge_type = "tx-tx"

    # Extract attention weights
    attention_df = extract_attention_df(attention_weights, gene_names_batch)

    # Visualize transcript-level attention differences
    print("Visualizing transcript-level attention differences...")
    layer_indices = [1, 2]  # 1-indexed layers in the dataframe
    head_indices = [1, 2]   # 1-indexed heads in the dataframe
    
    # # Visualize differences between layers for each head
    # visualize_attention_difference(
    #     attention_df=attention_df,
    #     edge_type=edge_type,
    #     compare_type='layers',
    #     layer_indices=layer_indices,
    #     head_indices=head_indices,
    #     gene_names=gene_names_batch
    # )
    
    # # Visualize differences between heads for each layer
    # visualize_attention_difference(
    #     attention_df=attention_df,
    #     edge_type=edge_type,
    #     compare_type='heads',
    #     layer_indices=layer_indices,
    #     head_indices=head_indices,
    #     gene_names=gene_names_batch
    # )

    # Gene-level visualization
    print("Computing gene-level attention patterns...")
    num_genes = len(transcripts['feature_name'].unique())
    gene_names = transcripts['feature_name'].unique().tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    layers = 5
    heads = 4
    
    attention_gene_matrix_dict = {
        "adj_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
        "count_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)]
    }
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # # Process each batch
    # for batch in dm.train[0:1]:
    #     with torch.no_grad():
    #         batch = batch.to(device)
    #         x_dict = batch.x_dict
    #         edge_index_dict = batch.edge_index_dict
    #         _, attention_weights = hetero_model(x_dict, edge_index_dict)
            
    #         transcript_ids = batch['tx'].id.cpu().numpy()
    #         gene_names = [id_to_gene[id] for id in transcript_ids]
            
    #         attention_df = extract_attention_df(attention_weights, gene_names)
            
    #         for layer_idx in range(layers):
    #             for head_idx in range(heads):
    #                 adj_matrix, count_matrix = summarize_attention_by_gene_df(
    #                     attention_df, 
    #                     layer_idx=layer_idx, 
    #                     head_idx=head_idx, 
    #                     edge_type='tx-tx', 
    #                     gene_to_idx=gene_to_idx, 
    #                     visualize=False
    #                 )
    #                 attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] += adj_matrix
    #                 attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx] += count_matrix

    # # Compute average attention weights
    # for layer_idx in range(layers):
    #     for head_idx in range(heads):
    #         attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] = safe_divide_sparse_numpy(
    #             attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx],
    #             attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx]
    #         )

    # # Save results
    # with open(results_dir / 'attention_gene_matrix_dict.pkl', 'wb') as f:
    #     pickle.dump(attention_gene_matrix_dict, f)
        
    # Load results
    with open(results_dir / 'attention_gene_matrix_dict969.pkl', 'rb') as f:
        attention_gene_matrix_dict = pickle.load(f)
    
    # # Generate comparison plots
    # compare_attention_patterns(
    #     attention_gene_matrix_dict,
    #     comparison_type='layers',
    #     edge_type=edge_type,
    #     top_k=15,
    #     gene_to_idx=gene_to_idx
    # )
    
    # compare_attention_patterns(
    #     attention_gene_matrix_dict,
    #     comparison_type='heads',
    #     edge_type=edge_type,
    #     top_k=15,
    #     gene_to_idx=gene_to_idx
    # )
    
    # # Visualize all attention patterns in a grid
    # visualize_all_attention_patterns(
    #     attention_gene_matrix_dict,
    #     edge_type=edge_type,
    #     gene_to_idx=gene_to_idx
    # )
    
    # # Visualize attention patterns with metrics
    # visualize_all_attention_patterns_with_metrics(
    #     attention_gene_matrix_dict,
    #     edge_type=edge_type,
    #     gene_to_idx=gene_to_idx
    # )
        
    # Visualize attention embeddings and patterns
    print("Visualizing attention embeddings and patterns...")
    
    # # Visualize all embeddings in a grid
    # visualize_all_embeddings(
    #     attention_gene_matrix_dict,
    #     method='umap',
    #     top_k_genes=20
    # )
    
    # Visualize average embedding
    visualize_average_embedding(
        attention_gene_matrix_dict,
        gene_names=gene_names,
        method='umap'
    )
    

if __name__ == '__main__':
    main() 