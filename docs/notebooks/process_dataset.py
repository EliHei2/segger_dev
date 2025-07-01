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
from docs.notebooks.visualization.utils import VisualizationConfig
from docs.notebooks.visualization.batch_visualization import extract_attention_df, visualize_attention_df
from docs.notebooks.visualization.gene_visualization import (
    visualize_all_attention_patterns, AttentionSummarizer
)
from docs.notebooks.visualization.utils import (
    safe_divide_sparse_numpy, get_top_genes_by_attention,
    VisualizationConfig, AttentionMatrixProcessor
)

from scipy.sparse import lil_matrix
import numpy as np

def main():
    # Configuration
    edge_type = "tx-bd"  # Changed from "tx-tx" to "tx-bd"
    max_cells = 10000  # Maximum number of cells to process
    
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
    
    # Get unique cell IDs and limit to max_cells
    all_cell_ids = cell_types['cell_id'].unique()
    if len(all_cell_ids) > max_cells:
        selected_cell_ids = []
        # Use the first max_cells cells
        for batch_idx, batch in enumerate(dm.train):
            cell_ids_batch = batch['bd'].id
            # add the cell ids to the selected_cell_ids
            selected_cell_ids = np.concatenate([selected_cell_ids, cell_ids_batch])
            # if the number of selected cells is greater than max_cells, break
            if len(selected_cell_ids) >= max_cells:
                print(f"The first {batch_idx} batches are used to select {max_cells} cells")
                max_batch_idx = batch_idx
                break
    else:
        selected_cell_ids = all_cell_ids
        print(f"Using all {len(selected_cell_ids)} cells")
    
    # Create cell ID to index mapping for selected cells
    cell_to_idx = {cell: idx for idx, cell in enumerate(selected_cell_ids)}
    cell_type_to_idx = {cell: idx for idx, cell in enumerate(cell_types['group'].unique())}
    
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

    # Extract attention weights
    attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, cell_types_dict = cell_types_dict, edge_type = edge_type)

    # Gene-level visualization
    print("Computing gene-level attention patterns...")
    num_genes = len(transcripts['feature_name'].unique())
    gene_names = transcripts['feature_name'].unique().tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    layers = 5
    heads = 4
    
    # Initialize attention matrix dictionary for both cases
    if edge_type == 'tx-tx':
        num_genes = len(gene_names)
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "count_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "gene_names": gene_names
        }
    elif edge_type == 'tx-bd':
        num_cells = len(selected_cell_ids)
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "cell_ids": selected_cell_ids.tolist(),
            "gene_names": gene_names
        }
    else:
        raise ValueError(f"Only tx-bd edge type is supported in this version")
    
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
        for batch_idx, batch in enumerate(dm.train[:max_batch_idx]):
            with torch.no_grad():
                print(f"Processing batch {batch_idx} of {max_batch_idx}")
                batch = batch.to(device)
                x_dict = batch.x_dict
                edge_index_dict = batch.edge_index_dict
                _, attention_weights = hetero_model(x_dict, edge_index_dict)
                
                transcript_ids = batch['tx'].id.cpu().numpy()
                gene_names_batch = [id_to_gene[id] for id in transcript_ids]
                
                cell_ids_batch = batch['bd'].id
                
                attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, cell_types_dict = cell_types_dict, edge_type = edge_type)
                
                for layer_idx in range(layers):
                    for head_idx in range(heads):
                        adj_matrix, _ = attention_summarizer.summarize_attention_by_gene_df(
                            attention_df, 
                            layer_idx=layer_idx, 
                            head_idx=head_idx
                        )
                        attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] += adj_matrix

        # Save results
        with open(results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl', 'wb') as f:
            pickle.dump(attention_gene_matrix_dict, f)
    else:
        print(f"Loading results from {results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl'}")
        # Load results
        with open(results_dir / f'attention_gene_matrix_dict_{edge_type}.pkl', 'rb') as f:
            attention_gene_matrix_dict = pickle.load(f)
    
    # select the genes in the gene_types_dict
    selected_genes = gene_types_dict.keys()
    selected_indices = [gene_to_idx[gene] for gene in selected_genes]
    
    # Create visualization configuration for tx-bd
    viz_config = VisualizationConfig(
        edge_type=edge_type,
        gene_types_dict=gene_types_dict,
        cell_to_idx=cell_to_idx,  # Use cell_to_idx instead of cell_type_to_idx
        cell_order=selected_cell_ids.tolist(),  # Use selected cell IDs
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
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Edge type: {edge_type}")
    print(f"Number of genes: {num_genes}")
    print(f"Number of cells: {len(selected_cell_ids)}")
    print(f"Number of layers: {layers}")
    print(f"Number of heads: {heads}")
    print(f"Matrix shape: {num_genes} x {len(selected_cell_ids)}")

if __name__ == '__main__':
    main() 