from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict_parquet import segment, load_model
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
import os
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUPY_CACHE_DIR"] = "./.cupy"
import cupy as cp
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask.dataframe as dd



seg_tag = "output-XETG00423__0052506__mng_04_TMA__20250310__160549"
model_version = 0
models_dir = Path("./models/project24_MNG_pqdm") / seg_tag


for s in os.listdir('data_tidy/pyg_datasets/project24_MNG_final/'):
    if s == seg_tag:

        XENIUM_DATA_DIR = Path(
            "data_tidy/pyg_datasets/project24_MNG_final") / s  
        


        benchmarks_dir = Path(
            "data_tidy/benchmarks/project24_MNG_final") / s 
        
        transcripts_file = Path(
        "/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/domenico_temp/xenium/xenium_output_files") / s / "transcripts.parquet"
        
        # Initialize the Lightning data module
        dm = SeggerDataModule(
            data_dir=XENIUM_DATA_DIR,
            batch_size=1,
            num_workers=1,
        )

        dm.setup()


        # Load in latest checkpoint
        model_path = models_dir / "lightning_logs" / f"version_{model_version}"
        model = load_model(model_path / "checkpoints")

        receptive_field = {"k_bd": 4, "dist_bd": 7.5, "k_tx": 5, "dist_tx": 3}

        segment(
            model,
            dm,
            save_dir=benchmarks_dir,
            seg_tag=s,
            transcript_file=transcripts_file,
            # file_format='anndata',
            receptive_field=receptive_field,
            min_transcripts=5,
            score_cut=0.75,
            # max_transcripts=1500,
            cell_id_col="segger_cell_id",
            use_cc=False,
            knn_method="kd_tree",
            verbose=True,
            gpu_ids=["0"],
            # client=client
        )
