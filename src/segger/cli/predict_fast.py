import click
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict_parquet import segment, load_model
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace
import os

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Path to default YAML configuration file
predict_yml = Path(__file__).parent / "configs" / "predict" / "default.yaml"

help_msg = "Run the Segger segmentation model."


@click.command(name="run_segmentation", help=help_msg)
@add_options(config_path=predict_yml)
@click.option("--segger_data_dir", type=Path, required=True, help="Directory containing the processed Segger dataset.")
@click.option("--models_dir", type=Path, required=True, help="Directory containing the trained models.")
@click.option("--benchmarks_dir", type=Path, required=True, help="Directory to save the segmentation results.")
@click.option("--transcripts_file", type=str, required=True, help="Path to the transcripts file.")
@click.option("--batch_size", type=int, default=1, help="Batch size for processing.")
@click.option("--num_workers", type=int, default=1, help="Number of workers for data loading.")
@click.option("--model_version", type=int, default=0, help="Model version to load.")
@click.option("--save_tag", type=str, default="segger_embedding_1001", help="Tag for saving segmentation results.")
@click.option("--min_transcripts", type=int, default=5, help="Minimum number of transcripts for segmentation.")
@click.option("--cell_id_col", type=str, default="segger_cell_id", help="Column name for cell IDs.")
@click.option("--use_cc", type=bool, default=False, help="Use connected components if specified.")
@click.option("--knn_method", type=str, default="cuda", help="Method for KNN computation.")
@click.option("--file_format", type=str, default="anndata", help="File format for output data.")
@click.option("--k_bd", type=int, default=4, help="K value for boundary computation.")
@click.option("--dist_bd", type=float, default=12.0, help="Distance for boundary computation.")
@click.option("--k_tx", type=int, default=5, help="K value for transcript computation.")
@click.option("--dist_tx", type=float, default=5.0, help="Distance for transcript computation.")
def run_segmentation(args: Namespace):

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing Segger data module...")
    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=args.segger_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup()

    logger.info("Loading the model...")
    # Load in the latest checkpoint
    model_path = Path(args.models_dir) / "lightning_logs" / f"version_{args.model_version}"
    model = load_model(model_path / "checkpoints")

    logger.info("Running segmentation...")
    segment(
        model,
        dm,
        save_dir=args.benchmarks_dir,
        seg_tag=args.save_tag,
        transcript_file=args.transcripts_file,
        file_format=args.file_format,
        receptive_field={"k_bd": args.k_bd, "dist_bd": args.dist_bd, "k_tx": args.k_tx, "dist_tx": args.dist_tx},
        min_transcripts=args.min_transcripts,
        cell_id_col=args.cell_id_col,
        use_cc=args.use_cc,
        knn_method=args.knn_method,
        verbose=True,
    )

    logger.info("Segmentation completed.")


if __name__ == "__main__":
    run_segmentation()
