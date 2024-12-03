import click
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
import time
import pandas as pd

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
    from segger.data.io import XeniumSample
    logging.info("Done.")

    # Determine mode to use
    if not args.mode in ['classic', 'experimental']:
        msg = (
            f"Argument 'mode' should be one of 'classic' or 'experimental'. "
            f"Got '{args.mode}'."
        )
        logging.error(msg)
        raise ValueError(msg)

    # Original (square tile)
    if args.mode == 'classic':
        logging.info("Initializing sample...")
        sample = XeniumSample(verbose=True)
        if args.gene_embedding_weights is not None:
            weights = pd.read_csv(args.gene_embedding_weights, index_col=0)
            sample.embedding_df = weights
        sample_dir = Path(args.sample_dir)
        sample.set_file_paths(
            transcripts_path=sample_dir / 'transcripts.parquet',
            boundaries_path=sample_dir / 'nucleus_boundaries.parquet',
        )
        sample.set_metadata()
        logging.info("Done.")

        logging.info("Saving dataset for Segger...")
        sample.save_dataset_for_segger(
            processed_dir=args.data_dir,
            x_size=args.tile_width,
            y_size=args.tile_height,
            d_x=args.tile_width - 20,
            d_y=args.tile_height - 20,
            margin_x=10,
            margin_y=10,
            compute_labels=True,
            k_tx=args.k_tx,
            r_tx=args.dist_tx,
            neg_sampling_ratio_approx=args.neg_sampling_ratio,
            sampling_rate=args.sampling_rate,
            val_prob=args.val_prob,
            test_prob=args.test_prob,
            workers=1,
        )
        logging.info("Dataset saved successfully.")

    # Fast dataset creation
    elif args.mode == 'experimental':
        logging.info("Initializing sample...")
        sample = STSampleParquet(
            base_dir=args.sample_dir,
            n_workers=args.n_workers,
            sample_type=args.sample_type,
        )
        if args.gene_embedding_weights is not None:
            weights = pd.read_csv(args.gene_embedding_weights, index_col=0)
            sample.set_transcript_embedding(weights)
        logging.info("Done.")

        logging.info("Saving dataset for Segger...")
        outs = sample.save(
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
