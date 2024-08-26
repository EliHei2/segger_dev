import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
import yaml


data_yml = Path(__file__).parent / 'configs' / 'create_dataset' / 'default.yaml'


help_msg = "Create Segger dataset from Xenium data"
@click.command(name="create_dataset", help=help_msg)
#@click.option('--foo', default="bar")  # add more options above, not below
@add_options(config_path=data_yml)
def create_dataset(args: Namespace):

    # Setup
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    from segger.data.utils import XeniumSample
    logging.info("Done.")

    # Load Xenium data
    logging.info("Loading data from Xenium sample...")
    sample = XeniumSample()
    xenium_dir = Path(args.xenium_dir)
    sample.load_transcripts(
        path=xenium_dir / 'transcripts.csv.gz',
        min_qv=args.min_qv,
    )
    sample.load_nuclei(xenium_dir / 'nucleus_boundaries.csv.gz')
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
    )
    logging.info("Done.")
