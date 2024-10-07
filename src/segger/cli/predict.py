import click
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import segment, load_model
from pathlib import Path
import logging
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@click.command(name="run_segmentation", help="Run the Segger segmentation model.")
@click.option('--segger_data_dir', type=Path, required=True, help='Directory containing the processed Segger dataset.')
@click.option('--models_dir', type=Path, required=True, help='Directory containing the trained models.')
@click.option('--benchmarks_dir', type=Path, required=True, help='Directory to save the segmentation results.')
@click.option('--transcripts_file', type=str, required=True, help='Path to the transcripts file.')
@click.option('--batch_size', type=int, default=1, help='Batch size for processing.')  
@click.option('--num_workers', type=int, default=1, help='Number of workers for data loading.')  
@click.option('--model_version', type=int, default=0, help='Model version to load.')  
@click.option('--save_tag', type=str, default='segger_embedding_1001_0.5', help='Tag for saving segmentation results.')  
@click.option('--min_transcripts', type=int, default=5, help='Minimum number of transcripts for segmentation.')  
@click.option('--cell_id_col', type=str, default='segger_cell_id', help='Column name for cell IDs.')  
@click.option('--use_cc', is_flag=True, default=False, help='Use connected components if specified.')  
@click.option('--knn_method', type=str, default='cuda', help='Method for KNN computation.')  
@click.option('--file_format', type=str, default='anndata', help='File format for output data.')  
@click.option('--k_bd', type=int, default=4, help='K value for boundary computation.')  
@click.option('--dist_bd', type=int, default=12, help='Distance for boundary computation.')  
@click.option('--k_tx', type=int, default=5, help='K value for transcript computation.')  
@click.option('--dist_tx', type=int, default=5, help='Distance for transcript computation.')  
def run_segmentation(segger_data_dir: Path, models_dir: Path, benchmarks_dir: Path, 
                     transcripts_file: str, batch_size: int = 1, num_workers: int = 1,
                     model_version: int = 0, save_tag: str = 'segger_embedding_1001_0.5',
                     min_transcripts: int = 5, cell_id_col: str = 'segger_cell_id',
                     use_cc: bool = False, knn_method: str = 'cuda',
                     file_format: str = 'anndata', k_bd: int = 4, dist_bd: int = 12,
                     k_tx: int = 5, dist_tx: int = 5):
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing Segger data module...")
    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=segger_data_dir,
        batch_size=batch_size,  
        num_workers=num_workers,  
    )
    
    dm.setup()
    
    logger.info("Loading the model...")
    # Load in the latest checkpoint
    model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
    model = load_model(model_path / 'checkpoints')

    logger.info("Running segmentation...")
    segment(
        model,
        dm,
        save_dir=benchmarks_dir,
        seg_tag=save_tag,
        transcript_file=transcripts_file,
        file_format=file_format,  
        receptive_field={'k_bd': k_bd, 'dist_bd': dist_bd, 'k_tx': k_tx, 'dist_tx': dist_tx},  
        min_transcripts=min_transcripts,
        cell_id_col=cell_id_col,
        use_cc=use_cc,
        knn_method=knn_method,
    )
    
    logger.info("Segmentation completed.")

if __name__ == '__main__':
    run_segmentation()
