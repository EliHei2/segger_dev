import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
import yaml

# Path to default YAML configuration file
data_yml = Path(__file__).parent / 'configs' / 'create_dataset' / 'default.yaml'

# CLI command to create a Segger dataset
help_msg = "Create Segger dataset from spatial transcriptomics data (Xenium or MERSCOPE)"
@click.command(name="create_dataset", help=help_msg)
@add_options(config_path=data_yml)
def create_dataset(args: Namespace):

    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import necessary packages based on the dataset type
    logging.info("Importing packages...")
    if args.dataset_type == 'xenium':
        from segger.data.io import XeniumSample as Sample
    elif args.dataset_type == 'merscope':
        from segger.data.io import MerscopeSample as Sample
    else:
        raise ValueError("Unsupported dataset type. Please choose 'xenium' or 'merscope'.")
    logging.info("Done.")

    # Load the dataset (Xenium or MERSCOPE)
    logging.info(f"Loading data from {args.dataset_type} sample...")
    sample = Sample()
    dataset_dir = Path(args.dataset_dir)

    # Load transcripts
    sample.load_transcripts(
        path=dataset_dir / args.transcripts_file,
        min_qv=args.min_qv,
        file_format=args.file_format,
    )

    # Load boundaries (nucleus boundaries for Xenium, cell boundaries for MERSCOPE)
    sample.load_boundaries(dataset_dir / args.boundaries_file, file_format=args.file_format)
    logging.info("Done.")

    # Save Segger dataset
    logging.info("Saving dataset for Segger...")
    data_dir = Path(args.data_dir)
    sample.save_dataset_for_segger(
        processed_dir=data_dir,
        x_size=args.x_size,
        y_size=args.y_size,
        d_x=args.d_x,
        d_y=args.d_y,
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        r_tx=args.r_tx,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
        compute_labels=args.compute_labels,
        sampling_rate=args.sampling_rate,
        num_workers=args.workers,
        receptive_field={
            "k_nc": args.k_nc,
            "dist_nc": args.dist_nc,
            "k_tx": args.k_tx,
            "dist_tx": args.dist_tx,
        },
        method=args.method,
        gpu=args.gpu
    )
    logging.info("Done.")
