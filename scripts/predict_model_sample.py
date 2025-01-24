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
import cupy as cp
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask.dataframe as dd


seg_tag = "bc_rep1_emb_final"
model_version = 6

seg_tag = "bc_fast_data_emb_major"
model_version = 1

segger_data_dir = Path('data_tidy/pyg_datasets') / seg_tag
models_dir = Path("./models") / seg_tag 
benchmarks_dir = Path("/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc")
transcripts_file = "data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet"
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,
    num_workers=0,
)

dm.setup()


# Load in latest checkpoint
model_path = models_dir / "lightning_logs" / f"version_{model_version}"
model = load_model(model_path / "checkpoints")

receptive_field = {"k_bd": 4, "dist_bd": 15, "k_tx": 5, "dist_tx": 3}

segment(
    model,
    dm,
    save_dir=benchmarks_dir,
    seg_tag=seg_tag,
    transcript_file=transcripts_file,
    # file_format='anndata',
    receptive_field=receptive_field,
    min_transcripts=5,
    score_cut=0.4,
    # max_transcripts=1500,
    cell_id_col="segger_cell_id",
    use_cc=False,
    knn_method="kd_tree",
    verbose=True,
    gpu_ids=["0"],
    # client=client
)
