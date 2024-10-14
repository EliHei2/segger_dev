import os
import sys
import argparse
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from segger.data.utils import SpatialTranscriptomicsDataset  # Updated dataset class
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from torch_geometric.nn import to_hetero
import warnings

os.environ["USE_PYGEOS"] = "0"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def check_and_create_raw_folder(directory):
    raw_dir = directory / "raw"
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            f"'{raw_dir}' does not exist. Creating this dummy folder because SpatialTranscriptomicsDataset requires it."
        )


def main(args):
    # CONFIG

    sys.path.insert(0, os.path.abspath("../.."))

    # Paths
    TRAIN_DIR = Path(args.train_dir)
    VAL_DIR = Path(args.val_dir)

    # Ensure 'raw' directories exist or create them with a warning
    check_and_create_raw_folder(TRAIN_DIR)
    check_and_create_raw_folder(VAL_DIR)

    # Load datasets using the new SpatialTranscriptomicsDataset class
    train_ds = SpatialTranscriptomicsDataset(root=TRAIN_DIR)
    val_ds = SpatialTranscriptomicsDataset(root=VAL_DIR)

    # Initialize model and convert to heterogeneous using to_hetero
    model = Segger(
        num_tx_tokens=args.num_tx_tokens,  # num_tx_tokens is now included
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=args.heads,
        num_mid_layers=args.mid_layers,  # mid_layers is now included
    )
    model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr=args.aggr)

    batch = train_ds[0]
    model.forward(batch.x_dict, batch.edge_index_dict)
    # Wrap the model in LitSegger
    litsegger = LitSegger(model=model)

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        devices=args.devices,
        max_epochs=args.epochs,
        default_root_dir=args.default_root_dir,
    )

    # DataLoaders for training and validation datasets
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_train, num_workers=0, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_val, num_workers=0, pin_memory=True, shuffle=False)

    # Train the model
    trainer.fit(litsegger, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Segger model")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to the validation data directory")
    parser.add_argument("--batch_size_train", type=int, default=4, help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=4, help="Batch size for validation")
    parser.add_argument(
        "--num_tx_tokens", type=int, default=500, help="Number of unique tx tokens for embedding"
    )  # num_tx_tokens default 500
    parser.add_argument("--init_emb", type=int, default=8, help="Initial embedding size")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels")
    parser.add_argument("--out_channels", type=int, default=16, help="Number of output channels")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument(
        "--mid_layers", type=int, default=1, help="Number of middle layers in the model"
    )  # mid_layers default 1
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument("--accelerator", type=str, default="cuda", help="Type of accelerator")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision mode")
    parser.add_argument("--devices", type=int, default=4, help="Number of devices")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--default_root_dir",
        type=str,
        default="./models/pancreas",
        help="Default root directory for logs and checkpoints",
    )

    args = parser.parse_args()
    main(args)
