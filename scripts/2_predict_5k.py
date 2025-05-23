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


XENIUM_DATA_DIR = Path( #raw data dir
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/xenium_seg_kit/human_CRC_real"
)
transcripts_file = (
   XENIUM_DATA_DIR / "transcripts.parquet"
)

SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/human_CRC_seg_nuclei") # preprocessed data dir


seg_tag = "human_CRC_seg_nuclei"
model_version = 0
models_dir = Path("./models") / seg_tag #trained model dir


output_dir = Path( #output dir
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/human_CRC_seg_nuclei"
)


# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=SEGGER_DATA_DIR,
    batch_size=1,
    num_workers=1,
)

dm.setup()


# Load in latest checkpoint
model_path = models_dir / "lightning_logs" / f"version_{model_version}"
model = load_model(model_path / "checkpoints")

receptive_field = {"k_bd": 4, "dist_bd": 15, "k_tx": 5, "dist_tx": 3}

segment(
    model,
    dm,
    save_dir=output_dir,
    seg_tag=seg_tag,
    transcript_file=transcripts_file,
    receptive_field=receptive_field,
    min_transcripts=5,
    score_cut=0.5,
    cell_id_col="segger_cell_id",
    save_transcripts= True,
    save_anndata= True,
    save_cell_masks= False,  # Placeholder for future implementation
    use_cc=False, # if one wants fragments (groups of similar transcripts not attached to any nuclei)
    knn_method="kd_tree",
    verbose=True,
    gpu_ids=["0"],
    # client=client
)
