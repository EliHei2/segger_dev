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


segger_data_dir = Path("./data_tidy/pyg_datasets/bc_embedding_1001")
models_dir = Path("./models/bc_embedding_1001_small")

dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=4,
    num_workers=2,
)

dm.setup()

metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
ls = LitSegger(
    num_tx_tokens=500,
    init_emb=8,
    hidden_channels=32,
    out_channels=8,
    heads=2,
    num_mid_layers=2,
    aggr="sum",
    metadata=metadata,
)

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator="cuda",
    strategy="auto",
    precision="16-mixed",
    devices=4,
    max_epochs=200,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)

batch = dm.train[0]
ls.forward(batch)


trainer.fit(model=ls, datamodule=dm)
