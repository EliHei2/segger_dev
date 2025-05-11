from segger.data.parquet.sample import STSampleParquet
from segger.training.segger_data_module import SeggerDataModule
from segger.training.train import LitSegger
from segger.models.segger_model import Segger
from torch_geometric.nn import to_hetero
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc


xenium_data_dir = Path('data_xenium')
segger_data_dir = Path('data_segger')
# Base directory to store Pytorch Lightning models
models_dir = Path('models')


sample = STSampleParquet(
    base_dir=xenium_data_dir,
    n_workers=4,
    sample_type='xenium', # this could be 'xenium_v2' in case one uses the cell boundaries from the segmentation kit.
    # weights=gene_celltype_abundance_embedding, # uncomment if gene-celltype embeddings are available
)




# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=2,
    num_workers=2,
)

dm.setup()

num_tx_tokens = 500

# If you use custom gene embeddings, use the following two lines instead:
# num_tx_tokens = dm.train[0].x_dict["tx"].shape[1] # Set the number of tokens to the number of genes


model = Segger(
    # is_token_based=is_token_based,
    num_tx_tokens=num_tx_tokens,
    init_emb=8,
    hidden_channels=64,
    out_channels=16,
    heads=4,
    num_mid_layers=3,
)
model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")

batch = dm.train[0]
model.forward(batch.x_dict, batch.edge_index_dict)
# Wrap the model in LitSegger
ls = LitSegger(model=model)

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator='cuda',
    strategy='auto',
    precision='16-mixed',
    devices=1, # set higher number if more gpus are available
    max_epochs=100,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)

# Fit model
trainer.fit(
    model=ls,
    datamodule=dm
)
