import os
import sys
import argparse
from pathlib import Path
import torch
import lightning as L
from torch_geometric.loader import DataLoader
from segger.data.utils import XeniumDataset
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from torch_geometric.nn import to_hetero
from lightning.pytorch.plugins.environments import LightningEnvironment


def main(args):
    # CONFIG
    os.environ["USE_PYGEOS"] = "0"
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    sys.path.insert(0, os.path.abspath("../.."))

    # Paths
    TRAIN_DIR = Path(args.train_dir)
    VAL_DIR = Path(args.val_dir)

    # Load datasets
    xe_train_ds = XeniumDataset(root=TRAIN_DIR)
    xe_val_ds = XeniumDataset(root=VAL_DIR)

    # Initialize model and trainer
    model = Segger(
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=args.heads,
    )
    model = to_hetero(
        model,
        (["tx", "nc"], [("tx", "belongs", "nc"), ("tx", "neighbors", "tx")]),
        aggr=args.aggr,
    )

    litsegger = LitSegger(model)
    trainer = L.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        devices=args.devices,
        max_epochs=args.epochs,
        default_root_dir=args.default_root_dir,
        plugins=[LightningEnvironment()],
    )

    # Train model
    train_loader = DataLoader(
        xe_train_ds,
        batch_size=args.batch_size_train,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        xe_val_ds,
        batch_size=args.batch_size_val,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    trainer.fit(litsegger, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Segger model")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to the training data directory",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to the validation data directory",
    )
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--batch_size_val",
        type=int,
        default=4,
        help="Batch size for validation",
    )
    parser.add_argument("--init_emb", type=int, default=8, help="Initial embedding size")
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels",
    )
    parser.add_argument("--out_channels", type=int, default=16, help="Number of output channels")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument("--accelerator", type=str, default="cuda", help="Type of accelerator")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision mode")
    parser.add_argument("--devices", type=int, default=4, help="Number of devices")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--default_root_dir",
        type=str,
        default="./models/MNG_big",
        help="Default root directory for logs and checkpoints",
    )

    args = parser.parse_args()
    main(args)
