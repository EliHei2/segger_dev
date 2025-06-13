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
from docs.notebooks.visualization.utils import VisualizationConfig, ComparisonConfig
from docs.notebooks.visualization.batch_visualization import extract_attention_df, visualize_attention_df
from docs.notebooks.visualization.gene_visualization import (
    visualize_all_attention_patterns, AttentionSummarizer
)
from docs.notebooks.visualization.gene_embedding import (
    visualize_attention_embedding, visualize_all_embeddings, visualize_average_embedding
)
from docs.notebooks.visualization.utils import (
    safe_divide_sparse_numpy, get_top_genes_by_attention,
    VisualizationConfig, ComparisonConfig, AttentionMatrixProcessor
)

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

    # Get classified genes (it contains the gene names and groups)
    gene_types = pd.read_csv(Path('data_xenium') / 'gene_groups_ordered_color.csv')
    gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
        
    # Get cell types
    cell_types = pd.read_csv(Path('data_xenium') / 'cell_groups.csv')
    cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
    
    # Get cell types order
    cell_types_order = pd.read_csv(Path('data_xenium') / 'cell_types_order_color.csv')
    cell_order = cell_types_order.sort_values('Order')['Cell Type'].tolist()
    
    # Create cell type to color mapping
    cell_type_to_color = dict(zip(cell_types_order['Cell Type'], cell_types_order['Color']))
    
    # Create gene type to color mapping
    gene_type_to_color = dict(zip(gene_types['group'], gene_types['Color']))
    
    cell_to_idx = {cell: idx for idx, cell in enumerate(cell_types['group'].unique())}
    
    # Get a sample batch ------------------------------------------------------------
    batch = dm.train[0].to(device)
    
    # Get gene names
    transcript_ids = batch['tx'].id.cpu().numpy()
    id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    gene_names_batch = [id_to_gene[id] for id in transcript_ids]
    cell_ids_batch = batch['bd'].id

    # Run forward pass to get attention weights
    with torch.no_grad():
        hetero_model = ls.model
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        _, attention_weights = hetero_model(x_dict, edge_index_dict)

    edge_type = "tx-tx"
    
    # Extract attention weights
    attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, cell_types_dict = cell_types_dict, edge_type = edge_type)

    # # Visualize transcript-level attention differences
    # print("Visualizing transcript-level attention differences...")
    # layer_indices = [0, 1, 2, 3, 4]  # 1-indexed layers in the dataframe
    # head_indices = [0, 1, 2, 3]   # 1-indexed heads in the dataframe
    
    # for layer_idx in layer_indices:
    #     for head_idx in head_indices:
    #         visualize_attention_df(attention_df, layer_idx, head_idx, edge_type, gene_types_dict=gene_types_dict, cell_types_dict=cell_types_dict)

    # Gene-level visualization
    print("Computing gene-level attention patterns...")
    num_genes = len(transcripts['feature_name'].unique())
    gene_names = transcripts['feature_name'].unique().tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    layers = 5
    heads = 4
    
    # Initialize attention matrix dictionary
    if edge_type == 'tx-tx':
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "count_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)]
        }
    elif edge_type == 'tx-bd':
        num_cells = len(cell_types['group'].unique())
        print(f"Number of cells: {num_cells}")
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "count_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)]
        }
    else:
        raise ValueError(f"Invalid edge type: {edge_type}")
    
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Process each batch ------------------------------------------------------------
    # If not file exists, process the batches and save the results
    if not (results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl').exists():
        # Initialize attention summarizer
        attention_summarizer = AttentionSummarizer(
            edge_type=edge_type,
            gene_to_idx=gene_to_idx,
            cell_to_idx=cell_to_idx
        )

        # Process each batch
        for batch_idx, batch in enumerate(dm.train):
            with torch.no_grad():
                print(f"Processing batch {batch_idx} of {len(dm.train)}")
                batch = batch.to(device)
                x_dict = batch.x_dict
                edge_index_dict = batch.edge_index_dict
                _, attention_weights = hetero_model(x_dict, edge_index_dict)
                
                transcript_ids = batch['tx'].id.cpu().numpy()
                id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
                gene_names_batch = [id_to_gene[id] for id in transcript_ids]
                
                cell_ids_batch = batch['bd'].id
                
                attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, cell_types_dict = cell_types_dict, edge_type = edge_type)
                
                for layer_idx in range(layers):
                    for head_idx in range(heads):
                        adj_matrix, count_matrix = attention_summarizer.summarize_attention_by_gene_df(
                            attention_df, 
                            layer_idx=layer_idx, 
                            head_idx=head_idx
                        )
                        attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] += adj_matrix
                        attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx] += count_matrix

        # Compute average attention weights
        for layer_idx in range(layers):
            for head_idx in range(heads):
                attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] = safe_divide_sparse_numpy(
                    attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx],
                    attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx]
                )

        # Save results
        with open(results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl', 'wb') as f:
            pickle.dump(attention_gene_matrix_dict, f)
    else:
        print(f"Loading results from {results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl'}")
        # Load results
        with open(results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl', 'rb') as f:
            attention_gene_matrix_dict = pickle.load(f)
    
    # # Load top genes names and indices
    # top_k = 20 # max: 50
    # with open(Path('intermediate_data') / f'top_genes_k50.pkl', 'rb') as f:
    #     top_genes, top_indices = pickle.load(f)
    
    # top_genes = top_genes[::-1][:top_k]
    # top_indices = top_indices[::-1][:top_k]
    
    # select the genes in the gene_types_dict
    selected_genes = gene_types_dict.keys()
    selected_indices = [gene_to_idx[gene] for gene in selected_genes]
    
    # Create visualization configuration
    viz_config = VisualizationConfig(
        edge_type=edge_type,
        gene_types_dict=gene_types_dict,
        cell_to_idx=cell_to_idx,
        cell_order=cell_order,
        cell_type_to_color=cell_type_to_color,
        gene_type_to_color=gene_type_to_color
    )
    
    # Visualize all attention patterns
    print("Visualizing attention patterns...")
    visualize_all_attention_patterns(
        attention_gene_matrix_dict,
        selected_gene_names=selected_genes,
        selected_gene_indices=selected_indices,
        config=viz_config
    )
    
    # Visualize attention embeddings and patterns
    if edge_type == 'tx-tx':
        print("Visualizing attention embeddings and patterns...")
        
        # add negative control genes to the gene_types_dict
        negative_control_genes = ['NegControlProbe_00034', 'NegControlProbe_00004', 'NegControlCodeword_0526', 'NegControlCodeword_0517', 'NegControlProbe_00035', 'NegControlCodeword_0514', 'NegControlCodeword_0506', 'NegControlCodeword_0534', 'NegControlProbe_00041', 'NegControlCodeword_0513', 'NegControlCodeword_0531', 'NegControlCodeword_0522', 'NegControlCodeword_0523', 'NegControlProbe_00022', 'NegControlProbe_00031', 'NegControlCodeword_0530', 'NegControlCodeword_0508', 'NegControlCodeword_0511', 'NegControlCodeword_0510', 'NegControlProbe_00024', 'NegControlProbe_00039', 'NegControlProbe_00002', 'NegControlCodeword_0528', 'NegControlCodeword_0540', 'NegControlCodeword_0503', 'NegControlCodeword_0536', 'NegControlCodeword_0539', 'NegControlProbe_00013', 'NegControlCodeword_0520', 'NegControlCodeword_0524', 'NegControlCodeword_0533', 'NegControlProbe_00042', 'BLANK_0069', 'NegControlProbe_00025', 'NegControlProbe_00017', 'NegControlCodeword_0502', 'NegControlProbe_00003', 'NegControlCodeword_0515', 'NegControlCodeword_0537', 'NegControlProbe_00012', 'NegControlProbe_00016', 'NegControlCodeword_0521', 'NegControlCodeword_0507', 'NegControlCodeword_0529', 'NegControlProbe_00033', 'NegControlCodeword_0505', 'NegControlCodeword_0519', 'NegControlCodeword_0509', 'NegControlCodeword_0500', 'NegControlCodeword_0538', 'NegControlProbe_00014', 'NegControlCodeword_0516', 'NegControlCodeword_0535', 'NegControlCodeword_0527', 'NegControlCodeword_0504', 'NegControlCodeword_0525', 'NegControlCodeword_0512', 'BLANK_0037', 'NegControlCodeword_0518', 'NegControlCodeword_0532', 'NegControlProbe_00019', 'BLANK_0006', 'NegControlCodeword_0501']
        
        gene_types_dict.update(dict(zip(negative_control_genes, ['negative_control'] * len(negative_control_genes))))
        
        # visualize the all attention patterns
        visualize_all_embeddings(
            attention_gene_matrix_dict,
            gene_names=gene_names,
            method='umap',
            gene_types_dict=gene_types_dict,
            gene_type_to_color=gene_type_to_color
        )
        
        # Visualize average embedding
        visualize_average_embedding(
            attention_gene_matrix_dict,
            gene_names=gene_names,
            method='umap',
            gene_types_dict=gene_types_dict,
            gene_type_to_color=gene_type_to_color
        )

if __name__ == '__main__':
    main() 