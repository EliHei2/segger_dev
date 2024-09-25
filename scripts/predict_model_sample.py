from segger.data.io import XeniumSample
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import predict, load_model
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


segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_0919')
models_dir = Path('./models/bc_embedding_0919')
benchmarks_path = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc')

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

# Perform segmentation (predictions)
segmentation_train = predict(
    model,
    dm.train_dataloader(),
    score_cut=0.5,  
    receptive_field=receptive_field,
    use_cc=True,
    # device='cuda',
    # num_workers=4
)

segmentation_val = predict(
    model,
    dm.val_dataloader(),
    score_cut=0.5,  
    receptive_field=receptive_field,
    use_cc=True,
    # use_cc=False,
    # device='cpu'
)

segmentation_test = predict(
    model,
    dm.test_dataloader(),
    score_cut=0.5,  
    receptive_field=receptive_field,
    use_cc=True,
    # use_cc=False,
    # device='cpu'
)



seg_combined = pd.concat([segmentation_train, segmentation_val, segmentation_test])
# Group by transcript_id and keep the row with the highest score for each transcript
seg_combined = pd.concat([segmentation_train, segmentation_val, segmentation_test]).reset_index()

# Group by transcript_id and keep the row with the highest score for each transcript
seg_final = seg_combined.loc[seg_combined.groupby('transcript_id')['score'].idxmax()]

# Drop rows where segger_cell_id is NaN
seg_final = seg_final.dropna(subset=['segger_cell_id'])

# Reset the index if needed
seg_final.reset_index(drop=True, inplace=True)

transcripts_df = dd.read_parquet('data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet')

# # Assuming seg_final is already computed with pandas
# # Convert seg_final to a Dask DataFrame to enable efficient merging with Dask
seg_final_dd = dd.from_pandas(seg_final, npartitions=transcripts_df.npartitions)

# # Step 1: Merge segmentation with the transcripts on transcript_id
# # Use 'inner' join to keep only matching transcript_ids
transcripts_df_filtered = transcripts_df.merge(seg_final_dd, on='transcript_id', how='inner')

# Compute the result if needed
transcripts_df_filtered = transcripts_df_filtered.compute()


from segger.data.utils import create_anndata
segger_adata = create_anndata(transcripts_df_filtered, cell_id_col='segger_cell_id')
segger_adata.write(benchmarks_path / 'adata_segger_embedding_full.h5ad')

