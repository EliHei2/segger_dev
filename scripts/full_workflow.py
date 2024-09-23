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

os.environ['DASK_DAEMON'] = 'False'

xenium_data_dir = Path('./data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1')
segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_0919')
models_dir = Path('./models/bc_embedding_0919')

scRNAseq_path = '/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad'

scRNAseq = sc.read(scRNAseq_path)

sc.pp.subsample(scRNAseq, 0.1)

# Step 1: Calculate the gene cell type abundance embedding
celltype_column = 'celltype_minor'
gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, celltype_column)




# Setup Xenium sample to create dataset
xs = XeniumSample(verbose=False) # , embedding_df=gene_celltype_abundance_embedding)
xs.set_file_paths(
    transcripts_path=xenium_data_dir / 'transcripts.parquet',
    boundaries_path=xenium_data_dir / 'nucleus_boundaries.parquet',
)
xs.set_metadata()
# xs.x_max = 1000
# xs.y_max = 1000

try:
    xs.save_dataset_for_segger(
        processed_dir=segger_data_dir,
        x_size=400,
        y_size=400,
        d_x=350,
        d_y=350,
        margin_x=20,
        margin_y=20,
        compute_labels=True,  # Set to True if you need to compute labels
        r_tx=5,
        k_tx=10,
        val_prob=0.4,
        test_prob=0.1,
        num_workers=6
    )
except AssertionError as err:
    print(f'Dataset already exists at {segger_data_dir}')

# # Base directory to store Pytorch Lightning models


# Initialize the Lightning model




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
    # use_cc=False,
    # device='cpu'
)

segmentation_test = predict(
    model,
    dm.test_dataloader(),
    score_cut=0.5,  
    receptive_field=receptive_field,
    # use_cc=False,
    # device='cpu'
)

batch = next(iter(dm.test_dataloader()))
batch = batch.to('cuda')
edge_index = get_edge_index(
    batch['bd'].pos[:, :2].cpu(),
    batch['tx'].pos[:, :2].cpu(),
    k=receptive_field['k_bd'],
    dist=receptive_field['dist_bd'],
    method='kd_tree',
).T


import pandas as pd
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
segger_adata.write(benchmarks_path / 'adata_segger_embedding.h5ad')

# transcripts_df_filtered.feature_name = transcripts_df_filtered.feature_name.apply(lambda x: x.decode('utf-8')).values

segger_adata = create_anndata(seg_final, cell_id_col='segger_cell_id')

# import torch

# x = batch['tx'].x
# x = torch.nan_to_num(x, nan = 0)
# is_one_dim = (x.ndim == 1) * 1
# # x = x[:, None]    
# x = ls.model.tx_embedding.tx(((x.sum(1) * is_one_dim).int())) * is_one_dim + ls.model.lin0.tx(x.float())  * (1 - is_one_dim) 
# # First layer
# x = x.relu()
# x = self.conv_first(x, edge_index) # + self.lin_first(x)
# x = x.relu()


metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
ls = LitSegger(
    num_tx_tokens=500,
    init_emb=8,
    hidden_channels=32,
    out_channels=8,
    heads=2,
    num_mid_layers=2,
    aggr='sum',
    metadata=metadata,
)

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator='cuda',  
    strategy='auto',
    precision='16-mixed',
    devices=2,  
    max_epochs=200,  
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)

batch = dm.train[0]
ls.forward(batch)


trainer.fit(
    model=ls,
    datamodule=dm
)

model_version = 2  # 'v_num' from training output above
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
metrics = pd.read_csv(model_path / 'metrics.csv', index_col=1)

fig, ax = plt.subplots(1,1, figsize=(2,2))

for col in metrics.columns.difference(['epoch']):
    metric = metrics[col].dropna()
    ax.plot(metric.index, metric.values, label=col)

ax.legend(loc=(1, 0.33))
ax.set_ylim(0, 1)
ax.set_xlabel('Step')