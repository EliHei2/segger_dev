import click
import os
import scanpy as sc
from segger.data.utils import calculate_gene_celltype_abundance_embedding
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
from segger.data.parquet.sample import STSampleParquet
from typing import Optional
import time

# Path to default YAML configuration file
data_yml = Path(__file__).parent / "configs" / "create_dataset" / "default_fast.yaml"

# CLI command to create a Segger dataset
help_msg = "Create Segger dataset from spatial transcriptomics data (Xenium or MERSCOPE)"


@click.command(name="create_dataset", help=help_msg)
@add_options(config_path=data_yml)
@click.option("--base_dir", type=Path, required=True, help="Directory containing the raw dataset.")
@click.option("--data_dir", type=Path, required=True, help="Directory to save the processed Segger dataset.")
@click.option(
    "--sample_type", type=str, default=None, help='The sample type of the raw data, e.g., "xenium" or "merscope".'
)
@click.option("--scrnaseq_file", type=Path, default=None, help="Path to the scRNAseq file.")
@click.option(
    "--celltype_column", type=str, default=None, help="Column name for cell type annotations in the scRNAseq file."
)
@click.option("--k_bd", type=int, default=3, help="Number of nearest neighbors for boundary nodes.")
@click.option("--dist_bd", type=float, default=15.0, help="Maximum distance for boundary neighbors.")
@click.option("--k_tx", type=int, default=3, help="Number of nearest neighbors for transcript nodes.")
@click.option("--dist_tx", type=float, default=5.0, help="Maximum distance for transcript neighbors.")
@click.option(
    "--tile_size",
    type=int,
    default=None,
    help="If provided, specifies the size of the tile. Overrides `tile_width` and `tile_height`.",
)
@click.option(
    "--tile_width", type=int, default=None, help="Width of the tiles in pixels. Ignored if `tile_size` is provided."
)
@click.option(
    "--tile_height", type=int, default=None, help="Height of the tiles in pixels. Ignored if `tile_size` is provided."
)
@click.option("--neg_sampling_ratio", type=float, default=5.0, help="Ratio of negative samples.")
@click.option("--frac", type=float, default=1.0, help="Fraction of the dataset to process.")
@click.option("--val_prob", type=float, default=0.1, help="Proportion of data for use for validation split.")
@click.option("--test_prob", type=float, default=0.2, help="Proportion of data for use for test split.")
@click.option("--n_workers", type=int, default=1, help="Number of workers for parallel processing.")
def create_dataset(args: Namespace):

    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # If scRNAseq file is provided, calculate gene-celltype embeddings
    gene_celltype_abundance_embedding = None
    if args.scrnaseq_file:
        logging.info("Calculating gene and celltype embeddings...")
        scRNAseq = sc.read(args.scrnaseq_file)
        sc.pp.subsample(scRNAseq, 0.1)
        gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, args.celltype_column)

    # Initialize the sample class
    logging.info("Initializing sample...")
    sample = STSampleParquet(
        base_dir=args.base_dir,
        n_workers=args.n_workers,
        sample_type=args.sample_type,
        weights=gene_celltype_abundance_embedding,
    )

    # Save Segger dataset
    logging.info("Saving dataset for Segger...")
    start_time = time.time()
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
        frac=args.frac,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
    )
    end_time = time.time()
    logging.info(f"Time to save dataset: {end_time - start_time} seconds")
    logging.info("Dataset saved successfully.")


if __name__ == "__main__":
    create_dataset()
