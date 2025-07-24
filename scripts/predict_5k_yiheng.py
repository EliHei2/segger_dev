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



seg_tag = "output-XETG00078__0041722__Region_1__20241203__142052"
model_version = 4
models_dir = Path("./models/MNG_5k_sampled/") / seg_tag


seg = "output-XETG00078__0041719__Region_2__20241203__142052"

XENIUM_DATA_DIR = Path(
    "/omics/odcf/analysis/OE0606_projects_temp/xenium_projects/20241209_Xenium5k_CNSL_BrM/20241209_Xenium5k_CNSL_BrM"
) / seg
SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/MNG_5k_sampled") / seg


benchmarks_dir = Path(
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks"
) / seg
transcripts_file = (
   XENIUM_DATA_DIR / "transcripts.parquet"
)
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=SEGGER_DATA_DIR,
    batch_size=2,
    num_workers=2,
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
    seg_tag=seg,
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
