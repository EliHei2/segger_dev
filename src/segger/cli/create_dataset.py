import click
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
from segger.data import XeniumSample, MerscopeSample
import time

# Path to default YAML configuration file
data_yml = Path(__file__).parent / "configs" / "create_dataset" / "default.yaml"

# CLI command to create a Segger dataset
help_msg = "Create Segger dataset from spatial transcriptomics data (Xenium or MERSCOPE)"


@click.command(name="create_dataset", help=help_msg)
@add_options(config_path=data_yml)
@click.option("--dataset_dir", type=Path, required=True, help="Directory containing the raw dataset.")
@click.option("--data_dir", type=Path, required=True, help="Directory to save the processed Segger dataset.")
@click.option("--sample_tag", type=str, required=True, help="Sample tag for the dataset.")
@click.option("--transcripts_file", type=str, required=True, help="Name of the transcripts file.")
@click.option("--boundaries_file", type=str, required=True, help="Name of the boundaries file.")
@click.option("--x_size", type=int, default=300, help="Size of each tile in x-direction.")
@click.option("--y_size", type=int, default=300, help="Size of each tile in y-direction.")
@click.option("--d_x", type=int, default=280, help="Tile overlap in x-direction.")
@click.option("--d_y", type=int, default=280, help="Tile overlap in y-direction.")
@click.option("--margin_x", type=int, default=10, help="Margin in x-direction.")
@click.option("--margin_y", type=int, default=10, help="Margin in y-direction.")
@click.option("--r_tx", type=int, default=5, help="Radius for computing neighborhood graph.")
@click.option("--k_tx", type=int, default=5, help="Number of nearest neighbors for the neighborhood graph.")
@click.option("--val_prob", type=float, default=0.1, help="Validation data split proportion.")
@click.option("--test_prob", type=float, default=0.2, help="Test data split proportion.")
@click.option("--neg_sampling_ratio", type=float, default=5, help="Ratio for negative sampling.")
@click.option("--sampling_rate", type=float, default=1, help="Sampling rate for the dataset.")
@click.option("--workers", type=int, default=1, help="Number of workers for parallel processing.")
@click.option("--gpu", is_flag=True, default=False, help="Use GPU if available.")
def create_dataset(
    args: Namespace,
    dataset_dir: Path,
    data_dir: Path,
    sample_tag: str,
    transcripts_file: str,
    boundaries_file: str,
    x_size: int,
    y_size: int,
    d_x: int,
    d_y: int,
    margin_x: int,
    margin_y: int,
    r_tx: int,
    k_tx: int,
    val_prob: float,
    test_prob: float,
    neg_sampling_ratio: float,
    sampling_rate: float,
    workers: int,
    gpu: bool,
):

    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Initialize the appropriate sample class based on dataset type
    logging.info("Initializing sample...")
    if args.dataset_type == "xenium":
        sample = XeniumSample()
    elif args.dataset_type == "merscope":
        sample = MerscopeSample()
    else:
        raise ValueError("Unsupported dataset type. Please choose 'xenium' or 'merscope'.")
    logging.info("Done.")

    # Set paths for transcripts and boundaries based on arguments
    logging.info(f"Setting file paths for {args.dataset_type} sample...")
    transcripts_path = dataset_dir / sample_tag / transcripts_file
    boundaries_path = dataset_dir / boundaries_file
    sample.set_file_paths(transcripts_path=transcripts_path, boundaries_path=boundaries_path)
    sample.set_metadata()
    logging.info("Done setting file paths.")

    # Save Segger dataset
    logging.info("Saving dataset for Segger...")
    start_time = time.time()
    sample.save_dataset_for_segger(
        processed_dir=data_dir,
        x_size=x_size,
        y_size=y_size,
        d_x=d_x,
        d_y=d_y,
        margin_x=margin_x,
        margin_y=margin_y,
        compute_labels=args.compute_labels,
        r_tx=r_tx,
        k_tx=k_tx,
        val_prob=val_prob,
        test_prob=test_prob,
        neg_sampling_ratio_approx=neg_sampling_ratio,
        sampling_rate=sampling_rate,
        num_workers=workers,
        gpu=gpu,
    )
    end_time = time.time()
    logging.info(f"Time to save dataset: {end_time - start_time} seconds")
    logging.info("Dataset saved successfully.")
