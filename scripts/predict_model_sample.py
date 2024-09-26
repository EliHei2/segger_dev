from segger.data.io import XeniumSample
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import segment, load_model
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning import Trainer
from pathlib import Path
from lightning.pytorch.plugins.environments import LightningEnvironment
from matplotlib import pyplot as plt
import seaborn as sns
# import pandas as pd
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import os
import dask.dataframe as dd
import pandas as pd
from pathlib import Path


segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_0919')
models_dir = Path('./models/bc_embedding_0919')
benchmarks_dir = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc')
transcripts_file = 'data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet'
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,  
    num_workers=2,  
)

dm.setup()


model_version = 2

# Load in latest checkpoint
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
model = load_model(model_path / 'checkpoints')
dm.setup()

receptive_field = {'k_bd': 4, 'dist_bd': 10,'k_tx': 5, 'dist_tx': 3}

segment(
    model,
    dm,
    save_dir=benchmarks_dir,
    seg_tag='test_segger_segment',
    transcript_file=transcripts_file,
    file_format='anndata',
    receptive_field = receptive_field,
    min_transcripts=10,
    max_transcripts=1000
    cell_id_col='segger_cell_id'
)