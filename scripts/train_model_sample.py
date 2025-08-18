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





segger_data_dir = Path("data_tidy/pyg_datasets/xe_bc_rep1_loss_emb2")
models_dir = Path("./models/xe_bc_rep1_loss_emb2")

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

# # If you use custom gene embeddings, use the following two lines instead:
is_token_based = False
num_tx_tokens = (
    dm.train[0].x_dict["tx"].shape[1]
)  # Set the number of tokens to the number of genes


model = Segger(
    num_tx_tokens=num_tx_tokens,
    # num_tx_tokens= 600,
    init_emb=16,
    hidden_channels=64,
    out_channels=16,
    heads=4,
    num_mid_layers=3,
)
model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="mean")

batch = dm.train[0]
model.forward(batch.x_dict, batch.edge_index_dict)
# Wrap the model in LitSegger
ls = LitSegger(model=model, align_loss=True, align_lambda=.5, cycle_length=10000)



# # Initialize the Lightning model
# ls = LitSegger(
#     # is_token_based=is_token_based,
#     num_tx_tokens= 7000,
#     init_emb=8,
#     hidden_channels=64,
#     out_channels=16,
#     heads=4,
#     num_mid_layers=3,
# )

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator="gpu",
    strategy="auto",
    precision="32",
    devices=1,  # set higher number if more gpus are available
    max_epochs=500,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)


trainer.fit(ls, datamodule=dm)
