from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import segment, get_similarity_scores, load_model, predict_batch, predict
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
import os
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import cupy as cp
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask.dataframe as dd

segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_1001')
models_dir = Path('./models/bc_embedding_1001_small')
benchmarks_dir = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc')
transcripts_file = 'data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet'
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,  
    num_workers=1,  
)

dm.setup()


model_version = 0

# Load in latest checkpoint
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
model = load_model(model_path / 'checkpoints')

receptive_field = {'k_bd': 4, 'dist_bd': 12,'k_tx': 5, 'dist_tx': 5}

segment(
    model,
    dm,
    save_dir=benchmarks_dir,
    seg_tag='segger_embedding_1001_0.5',
    transcript_file=transcripts_file,
    file_format='anndata',
    receptive_field = receptive_field,
    min_transcripts=5,
    # max_transcripts=1500,
    cell_id_col='segger_cell_id',
    use_cc=False,
    knn_method='cuda'
)