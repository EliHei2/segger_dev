import click
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
import time

# TODO: Add arguments: compute_labels, margin, x_min, x_max, y_min, y_max

# Path to default YAML configuration file
data_yml = Path(__file__).parent / 'configs' / 'create_dataset' / 'default.yaml'

@click.command(
    name="create_dataset",
    help="Create Segger dataset from spatial transcriptomics data",
)
#@click.option('--foo', default="bar")  # add more options above, not below
@add_options(config_path=data_yml)
def create_dataset(args: Namespace):
    '''
    CLI command to create a Segger dataset.
    '''
    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    logging.info("Importing packages...")
    from segger.data.parquet.sample import STSampleParquet
    logging.info("Done.")

    logging.info("Initializing sample...")
    sample = STSampleParquet(
        sample_dir=args.sample_dir,
        n_workers=args.n_workers,
        sample_type=args.sample_type
    )
    logging.info("Done.")

    logging.info("Saving dataset for Segger...")
    sample.save(
        data_dir=args.data_dir,
        k_bd=args.k_bd,
        dist_bd=args.dist_bd,
        k_tx=args.k_tx,
        dist_tx=args.dist_tx,
        tile_size=args.tile_size,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        neg_sampling_ratio=args.neg_sampling_ratio,
        frac=args.sampling_rate,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
    )
    logging.info("Dataset saved successfully.")
