from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import segment, get_similarity_scores, load_model, predict_batch
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
import os
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_0919')
models_dir = Path('./models/bc_embedding_0919')
benchmarks_dir = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc')
transcripts_file = 'data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet'
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,  
    num_workers=1,  
)

dm.setup()


model_version = 2

# Load in latest checkpoint
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
model = load_model(model_path / 'checkpoints')

# batch = next(iter(dm.train_dataloader())).to('cuda')
# print(batch)
# outs = model.model(batch.x_dict, batch.edge_index_dict)


# def get_similarity_scores(
#     model: torch.nn.Module, 
#     batch: Batch,
#     from_type: str,
#     to_type: str,
#     receptive_field: dict
# )


# def predict_batch(
#     lit_segger: torch.nn.Module,
#     batch: Batch,
#     score_cut: float,
#     receptive_field: Dict[str, float],
#     use_cc: bool = True,
#     knn_method: str = 'cuda'
# ) -> pd.DataFrame:

for batch in dm.train_dataloader():
    batch = batch.to('cuda')
    # outs = get_similarity_scores(
    #     model= model.model
    #     batch=batch,
    #     from_type='tx',
    #     to_type='bd',
    #     receptive_field={'k_bd': 4, 'dist_bd': 10,'k_tx': 5, 'dist_tx': 3}
    # )
    outs = predict_batch(
        lit_segger=model,
        batch=batch,
        score_cut=.5,
        receptive_field={'k_bd': 4, 'dist_bd': 10,'k_tx': 5, 'dist_tx': 3},
        use_cc = False,
        knn_method= 'cuda'
    )
    print(outs)
    
# dm.setup()

# receptive_field = {'k_bd': 4, 'dist_bd': 10,'k_tx': 5, 'dist_tx': 3}

# segment(
#     model,
#     dm,
#     save_dir=benchmarks_dir,
#     seg_tag='test_segger_segment',
#     transcript_file=transcripts_file,
#     file_format='anndata',
#     receptive_field = receptive_field,
#     min_transcripts=10,
#     max_transcripts=1000,
#     cell_id_col='segger_cell_id',
#     use_cc=False,
#     knn_method='kd_tree'
# )