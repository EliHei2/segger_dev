import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace

# Path to default YAML configuration file
train_yml = Path(__file__).parent / "configs" / "train" / "default.yaml"

help_msg = "Train the Segger segmentation model."


@click.command(name="train_model", help=help_msg)
@add_options(config_path=train_yml)
@click.option("--dataset_dir", type=Path, required=True, help="Directory containing the processed Segger dataset.")
@click.option(
    "--models_dir", type=Path, required=True, help="Directory to save the trained model and the training logs."
)
@click.option("--sample_tag", type=str, required=True, help="Sample tag for the dataset.")
@click.option("--init_emb", type=int, default=8, help="Size of the embedding layer.")
@click.option("--hidden_channels", type=int, default=32, help="Size of hidden channels in the model.")
@click.option("--num_tx_tokens", type=int, default=500, help="Number of transcript tokens.")
@click.option("--out_channels", type=int, default=8, help="Number of output channels.")
@click.option("--heads", type=int, default=2, help="Number of attention heads.")
@click.option("--num_mid_layers", type=int, default=2, help="Number of mid layers in the model.")
@click.option("--batch_size", type=int, default=4, help="Batch size for training.")
@click.option("--num_workers", type=int, default=2, help="Number of workers for data loading.")
@click.option(
    "--accelerator", type=str, default="cuda", help='Device type to use for training (e.g., "cuda", "cpu").'
)  # Ask for accelerator
@click.option("--max_epochs", type=int, default=200, help="Number of epochs for training.")
@click.option("--devices", type=int, default=4, help="Number of devices (GPUs) to use.")
@click.option("--strategy", type=str, default="auto", help="Training strategy for the trainer.")
@click.option("--precision", type=str, default="16-mixed", help="Precision for training.")
def train_model(args: Namespace):

    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    from segger.training.train import LitSegger
    from segger.training.segger_data_module import SeggerDataModule
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_lightning import Trainer

    logging.info("Done.")

    # Load datasets
    logging.info("Loading Xenium datasets...")
    dm = SeggerDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,  # Hard-coded batch size
        num_workers=args.num_workers,  # Hard-coded number of workers
    )

    dm.setup()
    logging.info("Done.")

    # Initialize model
    logging.info("Initializing Segger model and trainer...")
    metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
    ls = LitSegger(
        num_tx_tokens=args.num_tx_tokens,
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,  # Hard-coded value
        heads=args.heads,  # Hard-coded value
        num_mid_layers=args.num_mid_layers,  # Hard-coded value
        aggr="sum",  # Hard-coded value
        metadata=metadata,
    )

    # Forward pass to initialize the model
    if args.devices > 1:
        batch = dm.train[0]
        ls.forward(batch)

    # Initialize the Lightning trainer
    trainer = Trainer(
        accelerator=args.accelerator,  # Directly use the specified accelerator
        strategy=args.strategy,  # Hard-coded value
        precision=args.precision,  # Hard-coded value
        devices=args.devices,  # Hard-coded value
        max_epochs=args.max_epochs,  # Hard-coded value
        default_root_dir=args.models_dir,
        logger=CSVLogger(args.models_dir),
    )

    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(model=ls, datamodule=dm)
    logging.info("Done.")


@click.command(name="slurm", help="Train on Slurm cluster")
@add_options(config_path=train_yml)
def train_slurm(args):
    train_model(args)


@click.group(help="Train the Segger model")
def train():
    pass


train.add_command(train_slurm)

if __name__ == "__main__":
    train_model()
