#!/usr/bin/env python
import os
from pathlib import Path
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict_parquet import segment, load_model
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run segmentation on Xenium data')
    parser.add_argument('--seg_tag', type=str, required=True, help='Segmentation tag for model')
    parser.add_argument('--sample_id', type=str, required=True, help='Sample ID to process')
    parser.add_argument('--model_version', type=int, default=0, help='Model version number')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID to use')
    parser.add_argument('--models_root', type=str, default="./models/project24_MNG_pqdm", help='Root directory for models')
    parser.add_argument('--data_root', type=str, default="data_tidy/pyg_datasets/project24_MNG_final", help='Root directory for input data')
    parser.add_argument('--output_root', type=str, default="data_tidy/benchmarks/project24_MNG_final", help='Root directory for the segmentation outputs')
    parser.add_argument('--transcripts_root', type=str, default="/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/domenico_temp/xenium/xenium_output_files", help='Root directory for transcript files')
    parser.add_argument('--min_transcripts', type=int, default=5, help='Minimum number of transcripts')
    parser.add_argument('--score_cut', type=float, default=0.75, help='Score cutoff threshold')
    
    args = parser.parse_args()

    # Set up paths
    models_dir = Path(args.models_root) / args.seg_tag
    XENIUM_DATA_DIR = Path(args.data_root) / args.sample_id
    benchmarks_dir = Path(args.output_root) / args.sample_id
    transcripts_file = Path(args.transcripts_root) / args.sample_id / "transcripts.parquet"

    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUPY_CACHE_DIR"] = "./.cupy"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=XENIUM_DATA_DIR,
        batch_size=1,
        num_workers=1,
    )
    dm.setup()

    # Load in latest checkpoint
    model_path = models_dir / "lightning_logs" / f"version_{args.model_version}"
    model = load_model(model_path / "checkpoints")

    # Segmentation parameters
    receptive_field = {"k_bd": 4, "dist_bd": 7.5, "k_tx": 5, "dist_tx": 3}

    # Run segmentation
    segment(
        model,
        dm,
        save_dir=benchmarks_dir,
        seg_tag=args.sample_id,
        transcript_file=transcripts_file,
        receptive_field=receptive_field,
        min_transcripts=args.min_transcripts,
        score_cut=args.score_cut,
        cell_id_col="segger_cell_id",
        use_cc=False,
        knn_method="kd_tree",
        verbose=True,
        gpu_ids=[args.gpu_id],
    )

if __name__ == "__main__":
    main()