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


segger_data_dir = segger_data_dir = Path('data_tidy/pyg_datasets/bc_rep1_emb_final_200')
models_dir = Path("./models/bc_rep1_emb_final_200")

# Base directory to store Pytorch Lightning models
# models_dir = Path('models')

# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,
    num_workers=2,
)

dm.setup()

# is_token_based = True
# num_tx_tokens = 500

# If you use custom gene embeddings, use the following two lines instead:
is_token_based = False
num_tx_tokens = dm.train[0].x_dict["tx"].shape[1] # Set the number of tokens to the number of genes


num_bd_features = dm.train[0].x_dict["bd"].shape[1]

# Initialize the Lightning model
ls = LitSegger(
    is_token_based = is_token_based,
    num_node_features = {"tx": num_tx_tokens, "bd": num_bd_features},
    init_emb=8,    
    hidden_channels=64,
    out_channels=16,
    heads=4,
    num_mid_layers=3,
    aggr='sum',
    learning_rate=1e-3
)

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator='cuda',
    strategy='auto',
    precision='16-mixed',
    devices=2, # set higher number if more gpus are available
    max_epochs=400,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)


trainer.fit(
    model=ls,
    datamodule=dm
)