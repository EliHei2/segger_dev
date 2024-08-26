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
    from segger.data.utils import XeniumDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import to_hetero
    from segger.training.train import LitSegger
    from lightning import Trainer
    logging.info("Done.")

    # Load datasets
    logging.info("Loading Xenium datasets...")
    trn_ds = XeniumDataset(root=Path(args.data_dir) / 'train_tiles')
    val_ds = XeniumDataset(root=Path(args.data_dir) / 'val_tiles')
    kwargs = dict(
        num_workers=0,
        pin_memory=True,
    )
    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch_size_train, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size_val, shuffle=False, **kwargs
    )
    logging.info("Done.")

    # Initialize model
    logging.info("Initializing Segger model and trainer...")
    metadata = (
        ["tx", "nc"], [("tx", "belongs", "nc"), ("tx", "neighbors", "tx")]
    )
    lit_segger = LitSegger(
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=args.heads,
        aggr=args.aggr,
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
    )
    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(
        model=lit_segger,
        train_dataloaders=trn_loader,
        val_dataloaders=val_loader,
    )
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