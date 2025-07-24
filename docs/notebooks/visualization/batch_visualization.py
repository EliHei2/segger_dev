import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def extract_attention_df(attention_weights, gene_names=None, cell_ids=None, cell_types_dict=None, edge_type='tx-tx'):
    """Extract attention weights into a structured dataset."""
    assert edge_type in ['tx-tx', 'tx-bd'], "Edge type must be 'tx-tx' or 'tx-bd'"
    
    data = []
    for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
        if edge_type == 'tx-tx':
            alpha_tensor = alpha['tx']
            edge_index = edge_index['tx']
        else:
            alpha_tensor = alpha['bd']
            edge_index = edge_index['bd']
        
        alpha_tensor = alpha_tensor.cpu().detach().numpy()
        edge_index = edge_index.cpu().detach().numpy()
        
        for head_idx in range(alpha_tensor.shape[1]):
            head_weights = alpha_tensor[:, head_idx]
            
            for i, (src, dst) in enumerate(edge_index.T):
                entry = {
                    'source': int(src),
                    'target': int(dst),
                    'edge_type': edge_type,
                    'layer': layer_idx + 1,
                    'head': head_idx + 1,
                    'attention_weight': float(head_weights[i])
                }
                
                if gene_names is not None:
                    entry['source_gene'] = gene_names[src]
                    if edge_type == 'tx-tx':
                        entry['target_gene'] = gene_names[dst]
                    else:
                        entry['target_cell_id'] = cell_ids[dst]
                        entry['target_cell'] = cell_types_dict[cell_ids[dst]]
                
                data.append(entry)
    
    return pd.DataFrame(data)