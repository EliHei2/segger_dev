import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging


def train_model(args):

    # Setup
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    from segger.data.parquet.pyg_dataset import STPyGDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import to_hetero
    from segger.training.train import LitSegger
    from segger.training.segger_data_module import SeggerDataModule
    from pytorch_lightning import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False
    logging.info("Done.")

    # Load datasets
    logging.info("Loading Xenium datasets...")
    dm = SeggerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size_train,
        num_workers=1, 
    )
    logging.info("Done.")

    # Initialize model
    logging.info("Initializing Segger model and trainer...")
    metadata = (
        ["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]
    )
    lit_segger = LitSegger(
        num_tx_tokens=args.num_tx_tokens,
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=args.heads,
        aggr=args.aggr,
        num_mid_layers=args.num_mid_layers,
        metadata=metadata,
    )

    # Initialize lightning trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        devices=args.devices,
        max_epochs=args.epochs,
        default_root_dir=args.model_dir,
        logger=CSVLogger(args.model_dir),
    )
    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(model=lit_segger, datamodule=dm)
    logging.info("Done...")


train_yml = Path(__file__).parent / 'configs' / 'train' / 'default.yaml'


@click.command(name="slurm", help="Train on Slurm cluster")
#@click.option('--foo', default="bar")  # add more options above, not below
@add_options(config_path=train_yml)
def train_slurm(args):
    train_model(args)


@click.group(help="Train the Segger model")
def train():
    pass


train.add_command(train_slurm)