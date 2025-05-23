from segger.training.segger_data_module import SeggerDataModule
# from segger.prediction.predict import predict, load_model
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from torch_geometric.nn import to_hetero
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer
from pathlib import Path
from lightning.pytorch.plugins.environments import LightningEnvironment
from matplotlib import pyplot as plt
import seaborn as sns
# import pandas as pd
from segger.data.utils import calculate_gene_celltype_abundance_embedding
# import scanpy as sc
import os
from lightning import LightningModule



segger_data_dir = Path("data_tidy/pyg_datasets/MNG_5k_sampled/output-XETG00078__0041719__Region_2__20241203__142052/")
models_dir = Path("./models/MNG_5k_sampled/output-XETG00078__0041719__Region_2__20241203__142052/")

# Base directory to store Pytorch Lightning models
# models_dir = Path('models')

# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=3,
    num_workers=3,
)

dm.setup()

# is_token_based = True
# num_tx_tokens = 500

# If you use custom gene embeddings, use the following two lines instead:
is_token_based = False
num_tx_tokens = (
    dm.train[0].x_dict["tx"].shape[1]
)  # Set the number of tokens to the number of genes


model = Segger(
    # is_token_based=is_token_based,
    num_tx_tokens= num_tx_tokens,
    init_emb=8,
    hidden_channels=32,
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
    accelerator="gpu",
    strategy="auto",
    precision="32",
    devices=4,  # set higher number if more gpus are available
    max_epochs=250,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)


trainer.fit(ls, datamodule=dm)
