import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging

help_msg = "Train the Segger segmentation model."

@click.command(name="train_model", help=help_msg)
@add_options(config_path=train_yml)
@click.option('--dataset_dir', type=Path, required=True, help='Directory containing the processed Segger dataset.')
@click.option('--models_dir', type=Path, required=True, help='Directory to save the trained model and the training logs.')
@click.option('--sample_tag', type=str, required=True, help='Sample tag for the dataset.')
@click.option('--init_emb', type=int, default=8, help='Size of the embedding layer.')
@click.option('--hidden_channels', type=int, default=32, help='Size of hidden channels in the model.')
@click.option('--num_tx_tokens', type=int, default=500, help='Number of transcript tokens.')
@click.option('--out_channels', type=int, default=8, help='Number of output channels.') 
@click.option('--heads', type=int, default=2, help='Number of attention heads.') 
@click.option('--num_mid_layers', type=int, default=2, help='Number of mid layers in the model.') 
@click.option('--batch_size', type=int, default=4, help='Batch size for training.') 
@click.option('--num_workers', type=int, default=2, help='Number of workers for data loading.') 
@click.option('--accelerator', type=str, default='cuda', help='Device type to use for training (e.g., "cuda", "cpu").')  # Ask for accelerator
@click.option('--max_epochs', type=int, default=200, help='Number of epochs for training.') 
@click.option('--devices', type=int, default=4, help='Number of devices (GPUs) to use.') 
@click.option('--strategy', type=str, default='auto', help='Training strategy for the trainer.') 
@click.option('--precision', type=str, default='16-mixed', help='Precision for training.') 
def train_model(dataset_dir: Path, models_dir: Path, sample_tag: str,
                init_emb: int = 8, hidden_channels: int = 32, num_tx_tokens: int = 500,
                out_channels: int = 8, heads: int = 2, num_mid_layers: int = 2,
                batch_size: int = 4, num_workers: int = 2, 
                accelerator: str = 'cuda', max_epochs: int = 200,
                devices: int = 4, strategy: str = 'auto', precision: str = '16-mixed'):

    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    from segger.data.io import XeniumSample
    from segger.training.train import LitSegger
    from segger.training.segger_data_module import SeggerDataModule
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_lightning import Trainer
    logging.info("Done.")

    # Load datasets
    logging.info("Loading Xenium datasets...")
    dm = SeggerDataModule(
        data_dir=dataset_dir,
        batch_size=batch_size,  # Hard-coded batch size
        num_workers=num_workers,  # Hard-coded number of workers
    )

    dm.setup()
    logging.info("Done.")

    # Initialize model
    logging.info("Initializing Segger model and trainer...")
    metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
    ls = LitSegger(
        num_tx_tokens=num_tx_tokens,
        init_emb=init_emb,
        hidden_channels=hidden_channels,
        out_channels=out_channels,  # Hard-coded value
        heads=heads,  # Hard-coded value
        num_mid_layers=num_mid_layers,  # Hard-coded value
        aggr='sum',  # Hard-coded value
        metadata=metadata,
    )

    # Initialize the Lightning trainer
    trainer = Trainer(
        accelerator=accelerator,  # Directly use the specified accelerator
        strategy=strategy,  # Hard-coded value
        precision=precision,  # Hard-coded value
        devices=devices,  # Hard-coded value
        max_epochs=max_epochs,  # Hard-coded value
        default_root_dir=models_dir,
        logger=CSVLogger(models_dir),
    )
    
    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(
        model=ls,
        datamodule=dm
    )
    logging.info("Done.")

train_yml = Path(__file__).parent / 'configs' / 'train' / 'default.yaml'

@click.command(name="slurm", help="Train on Slurm cluster")
@add_options(config_path=train_yml)
def train_slurm(args):
    train_model(args)

@click.group(help="Train the Segger model")
def train():
    pass

train.add_command(train_slurm)
